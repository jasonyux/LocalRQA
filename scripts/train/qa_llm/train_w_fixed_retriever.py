from configs.retriever_config import SEARCH_CONFIG
from transformers import (
    set_seed,
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    HfArgumentParser,
)
from typing import Union
from src.trainer.e2e_trainer import FixedRetrieverTrainer
from src.trainer.training_arguments import E2EQATrainingArguments
from src.trainer.data_utils import GroundedQADataset, NoopDataCollator
from src.evaluation.evaluator import EvaluatorConfig
from src.model.wrappers import QAFromLlamaModel
from dataclasses import dataclass, field
import wandb
import os
import sys
import json


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/opt-iml-1.3b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    search_algo: str = field(
        default="inner_product",
        metadata={"help": "The search algorithm to use"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dset: str = field(
        # default="data/training/word_sorting/ws_self_improve.jsonl",  # please do supply, in case accidens happen
        default='',
        metadata={"help": "Path to training dataset"}
    )
    eval_dset: str = field(
        # default="data/validation/word_sorting/ws_self_improve_val.jsonl",
        default='',
        metadata={"help": "Path to vallidation dataset"}
    )
    end_data_idx: Union[int, None] = field(
        default=None,
        metadata={"help": "The index of the last data point to use"}
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the data"}
    )
    eval_model_wrapper_cls: str = field(
        default="self-improve",
        metadata={"help": "The class to use for the eval model wrapper"},
    )

    def __post_init__(self):
        if self.train_dset == '' or self.eval_dset == '':
            raise ValueError("Need both a training/validation file.")
        return


@dataclass
class LoggerArguments:
    """
    Arguments pertaining to using wandb for logging
    """

    run_group: str = field(
        default="debug",
        metadata={"help": "wandb run group"}
    )


def to_dataset(data_args, tokenizer):
    return


def main(model_args: ModelArguments, data_args: DataArguments, logger_args: LoggerArguments, training_args: E2EQATrainingArguments):
    set_seed(training_args.seed)

    model_name: str = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding=True, truncation=True, return_tensors="pt"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.eos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dset, eval_dset = to_dataset(data_args, tokenizer)

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
    
    search_kwargs = SEARCH_CONFIG[model_args.search_algo]
    eval_config = EvaluatorConfig(  # type: ignore
        gen_f1 = True,
        gen_precision = True,
        gen_rouge = True,
        gen_latency = True,
    )

    trainer = FixedRetrieverTrainer(
        retriever_model= None,
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
    return


if __name__ == "__main__":
    raise NotImplementedError("This script is not ready for use yet.")
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
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, 'all_args.json'), 'w') as f:
        all_args = {
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict()
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    # train
    main(model_args, data_args, logger_args, training_args)