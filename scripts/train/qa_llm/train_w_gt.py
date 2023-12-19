from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    HfArgumentParser
)
from dataclasses import dataclass, field
from open_rqa.trainers.qa_llm.datasets import SupervisedRQADataset
from open_rqa.trainers.qa_llm.arguments import E2EQATrainingArguments
from open_rqa.trainers.utils import (
    remove_optimizer_weights,
    init_logger,
    create_dir_if_not_exists
)
import open_rqa.trainers.dist_utils as dist_utils
import pickle
import wandb
import random
import logging
import sys
import json
import os


logger: logging.Logger


@dataclass
class LoggerArguments:
    """
    Arguments pertaining to using wandb for logging
    """

    run_group: str = field(
        default="debug",
        metadata={"help": "wandb run group"}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="lmsys/vicuna-13b-v1.5-16k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: str = field(
        default="data/training/databricks_clean/train_w_a.pkl",
        metadata={"help": "Path for cached train dataset"},
    )
    eval_file: str = field(
        default='data/training/databricks_clean/eval_w_a.pkl',
        metadata={"help": "Path for cached eval dataset"},
    )
    test_file: str = field(
        default='data/training/databricks_clean/test_w_a.pkl',
        metadata={"help": "Path for cached test dataset"},
    )


def main(model_args: ModelArguments, data_args: DataArguments, logger_args: LoggerArguments, training_args: E2EQATrainingArguments):
    random.seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    with open(data_args.train_file, 'rb') as fread:
        train_data = pickle.load(fread)
    with open(data_args.eval_file, 'rb') as fread:
        eval_data = pickle.load(fread)
    with open(data_args.test_file, 'rb') as fread:
        test_data = pickle.load(fread)

    train_dset = SupervisedRQADataset(
        qa_w_doc_data=train_data,
        tokenizer=tokenizer,
        end_data_idx=None,
        shuffle=True
    )
    eval_dset = SupervisedRQADataset(
        qa_w_doc_data=eval_data,
        tokenizer=tokenizer,
        end_data_idx=None,
        shuffle=False
    )
    test_dset = SupervisedRQADataset(
        qa_w_doc_data=test_data,
        tokenizer=tokenizer,
        end_data_idx=None,
        shuffle=False
    )

    if 'wandb' in training_args.report_to:
        if dist_utils.is_main():
            run = wandb.init(
                id=wandb.util.generate_id(),
                project='tamarin',
                entity='tamarin',
                name=training_args.output_dir.split("/")[-1] or None,
                group='databricks_gen',
                config=all_args,
                resume="allow",
            )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dset,
    #     eval_dataset=eval_dset,
    #     data_collator=default_data_collator,
    #     tokenizer=tokenizer,
    # )
    ###
    # if it is already initialized, huggingface will use it
    all_args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'logger_args': vars(logger_args),
        'training_args': training_args.to_dict()
    }
    if 'wandb' in training_args.report_to:
        run = wandb.init(
            project='tamarin',
            entity='tamarin',
            name=training_args.output_dir.split("/")[-1] or None,
            group=logger_args.run_group,
            config=all_args,
        )
    
    eval_config = EvaluatorConfig(  # type: ignore
        gen_f1 = True,
        gen_precision = True,
        gen_rouge = True,
        gen_latency = True,
    )

    # TODO: fix this to not use retriever as it's supervised
    trainer = FixedRetrieverTrainer(
        retriever_model=None,
        model=model,
        args=training_args,
        eval_config=eval_config,
        eval_wrapper_class=QAFromLlamaModel,
        eval_search_kwargs=search_kwargs,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        data_collator=NoopDataCollator(),
        tokenizer=tokenizer,
    )
    trainer.train()

    if dist_utils.is_main():
        remove_optimizer_weights(training_args.output_dir)

    # test
    trainer.predict(test_dset)

    if 'wandb' in training_args.report_to:
        wandb.finish()
    return


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, DataArguments, LoggerArguments, E2EQATrainingArguments),
        description="E2E QA training script"
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, logger_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, logger_args, training_args = parser.parse_args_into_dataclasses()
    print('received model_args:')
    print(json.dumps(vars(model_args), indent=2, sort_keys=True))
    print('received data_args:')
    print(json.dumps(vars(data_args), indent=2, sort_keys=True))
    print('received logger_args:')
    print(json.dumps(vars(logger_args), indent=2, sort_keys=True))
    print('received training_args:')
    print(json.dumps(training_args.to_dict(), indent=2, sort_keys=True))
    
    # save config to model_args.model_save_path
    create_dir_if_not_exists(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'all_args.json'), 'w') as f:
        all_args = {
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict()
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    
    logger = init_logger(is_main=True, filename=os.path.join(training_args.output_dir, 'train.log'))

    main(model_args, data_args, logger_args, training_args)