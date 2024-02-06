import sys
from typing import List, Dict, Type
import pickle
import argparse
import os
import json
import jsonlines
import numpy as np

from transformers import (
    AutoTokenizer, AutoModel,
    HfArgumentParser
)
import torch
from torch.utils.data import DataLoader, SequentialSampler
import wandb

from open_rqa.trainers.utils import (
    init_logger,
    create_dir_if_not_exists
)
from open_rqa.trainers.retriever.arguments import ModelArguments, DataArguments, FidTrainingArgs, RetrievalQATrainingArguments, LoggerArguments
from open_rqa.qa_llms.fid import FiDT5
from open_rqa.trainers.retriever.datasets import FidDataset, FidCollator, RetrieverCollator, load_fid_data
from open_rqa.trainers.retriever.retriever_fid_trainer import FidRetrieverTrainer, EvaluatorConfig


def evaluate(model, dataset, dataloader, tokenizer):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    model.overwrite_forward_crossattention()
    model.reset_score_storage() 
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch
            model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_new_tokens=50
            )
            crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]

                for j in range(context_ids.size(1)):
                    example['ctxs'][j]['crossattention_score'] = crossattention_scores[k, j].item()


def get_crossattention_score(model_args, data_file, fid_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(fid_args.reader_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator_function = FidCollator(fid_args.text_maxlength, tokenizer)
    eval_examples = load_fid_data(data_file)
    eval_dataset = FidDataset(
        eval_examples, 
        fid_args.n_context,
        score_key="score"
    )
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=fid_args.reader_batch_size,
        collate_fn=collator_function
    )
    
    model = FiDT5.from_t5(
        fid_args.reader_model_path,
    )
    model.to("cuda")

    evaluate(model, eval_dataset, eval_dataloader, tokenizer)

    return eval_dataset.data

def modify_score(examples):
    # This function assign the gold document with "max+std" score
    for sample in examples:
        all_scores = []
        for doc in sample['ctxs']:
            if doc.get('crossattention_score', None):
                all_scores.append(doc['crossattention_score'])
        
        mean = np.mean(all_scores)
        std = np.std(all_scores)
        max = np.max(all_scores)
        sample['ctxs'][0]['crossattention_score'] = max + std # The first document is the gold one


def main(model_args, data_args, fid_args, training_args, logger_args):
    # Get cross attention score based on the reader
    if not fid_args.with_score:
        train_examples = get_crossattention_score(model_args, data_args.train_file, fid_args, training_args)
        eval_examples = get_crossattention_score(model_args, data_args.eval_file, fid_args, training_args)

        # Save the data
        train_filepath_wscore = data_args.train_file.split(".json")[0] + "_wscore.json"
        with open(train_filepath_wscore, 'w') as fout:
            json.dump(train_examples, fout, indent=4)

        eval_filepath_wscore = data_args.eval_file.split(".json")[0] + "_wscore.json"
        with open(eval_filepath_wscore, 'w') as fout:
            json.dump(eval_examples, fout, indent=4)

        # if there is gold document, manually modify the cross attention score for the gold document
        modify_score(train_examples)
        modify_score(eval_examples)
    else:
        # load the train and eval data with score
        train_examples = json.load(open(data_args.train_file, "r"))
        eval_examples = json.load(open(data_args.eval_file, "r"))

        # if there is gold document, manually modify the cross attention score for the gold document
        modify_score(train_examples)
        modify_score(eval_examples)

        
    #Load data
    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    collator_function = RetrieverCollator(
        tokenizer
    )
    train_dataset = FidDataset(train_examples, fid_args.n_context, score_key="crossattention_score")
    eval_dataset = FidDataset(eval_examples, fid_args.n_context, score_key="crossattention_score")


    eval_config = EvaluatorConfig(
        gen_latency = False,
        batch_size = training_args.per_device_eval_batch_size
    )

    all_args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'training_args': training_args.to_dict(),
        'fid_args': vars(fid_args),
        'logger_args': vars(logger_args),
    }
    if 'wandb' in training_args.report_to:
        run = wandb.init(
            project=logger_args.run_project,
            entity=logger_args.run_entity,
            name=training_args.output_dir.split("/")[-1] or None,
            group=logger_args.run_group,
            config=all_args,
        )

    trainer = FidRetrieverTrainer(
        model=model,
        training_args=training_args,
        data_args=data_args,
        fid_args=fid_args,
        eval_config=eval_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_function,
        tokenizer=tokenizer,
    )

    if training_args.do_eval and not training_args.do_train:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, DataArguments, FidTrainingArgs, RetrievalQATrainingArguments, LoggerArguments),
        description="QA Retriever training script"
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, fid_args, training_args, logger_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, fid_args, training_args, logger_args = parser.parse_args_into_dataclasses()


    print('received model_args:')
    print(json.dumps(vars(model_args), indent=2, sort_keys=True))
    print('received data_args:')
    print(json.dumps(vars(data_args), indent=2, sort_keys=True))
    print('received logger_args:')
    print(json.dumps(vars(logger_args), indent=2, sort_keys=True))
    print('received training_args:')
    print(json.dumps(training_args.to_dict(), indent=2, sort_keys=True))
    print('received fid_args:')
    print(json.dumps(vars(fid_args), indent=2, sort_keys=True))
    
    # save config to model_args.model_save_path
    create_dir_if_not_exists(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'all_args.json'), 'w', encoding='utf-8') as f:
        all_args = {
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict(),
            'fid_args': vars(fid_args)
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    
    logger = init_logger(is_main=True, filename=os.path.join(training_args.output_dir, 'train.log'))

    main(model_args, data_args, fid_args, training_args, logger_args)
