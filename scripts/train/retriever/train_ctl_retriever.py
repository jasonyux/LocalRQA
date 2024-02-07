import sys
import os
import json
import jsonlines

from transformers import (
    AutoTokenizer, AutoModel,
    HfArgumentParser
)
import wandb

from local_rqa.trainers.utils import (
    init_logger,
    create_dir_if_not_exists
)
from local_rqa.trainers.retriever.arguments import LoggerArguments, ModelArguments, DataArguments, ContrasitiveTrainingArgs, RetrievalQATrainingArguments
from local_rqa.trainers.retriever.datasets import ContrastiveRetrievalDataset, NoopDataCollator
from local_rqa.trainers.retriever.retriever_trainer import RetrieverTrainer, EvaluatorConfig


def main(model_args, data_args, contrastive_args, training_args, logger_args):
    with jsonlines.open(data_args.train_file) as fread:
        train_data = list(fread)
    with jsonlines.open(data_args.eval_file) as fread:
        eval_data = list(fread)

    train_dataset = ContrastiveRetrievalDataset(
        train_data, shuffle=True
    )
    eval_dataset = ContrastiveRetrievalDataset(
        eval_data, shuffle=True
    )


    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    eval_config = EvaluatorConfig(
        gen_latency = False,
        batch_size = training_args.per_device_eval_batch_size
    )

    all_args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'training_args': training_args.to_dict(),
        'contrastive_args': vars(contrastive_args),
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

    trainer = RetrieverTrainer(
        model=model,
        training_args=training_args,
        data_args=data_args,
        contrastive_args=contrastive_args,
        eval_config=eval_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=NoopDataCollator(),
        tokenizer=tokenizer,
    )

    if training_args.do_eval and not training_args.do_train:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, DataArguments, ContrasitiveTrainingArgs, RetrievalQATrainingArguments, LoggerArguments),
        description="QA Retriever training script"
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, contrastive_args, training_args, logger_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, contrastive_args, training_args, logger_args = parser.parse_args_into_dataclasses()

    print('received model_args:')
    print(json.dumps(vars(model_args), indent=2, sort_keys=True))
    print('received data_args:')
    print(json.dumps(vars(data_args), indent=2, sort_keys=True))
    print('received logger_args:')
    print(json.dumps(vars(logger_args), indent=2, sort_keys=True))
    print('received training_args:')
    print(json.dumps(training_args.to_dict(), indent=2, sort_keys=True))
    print('received contrastive_args:')
    print(json.dumps(vars(contrastive_args), indent=2, sort_keys=True))
    
    # save config to model_args.model_save_path
    create_dir_if_not_exists(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'all_args.json'), 'w', encoding='utf-8') as f:
        all_args = {
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict(),
            'contrastive_args': vars(contrastive_args)
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    
    logger = init_logger(is_main=True, filename=os.path.join(training_args.output_dir, 'train.log'))

    main(model_args, data_args, contrastive_args, training_args, logger_args)
