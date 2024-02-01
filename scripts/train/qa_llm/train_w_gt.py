from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    HfArgumentParser
)
from accelerate.utils import DistributedType
from dataclasses import dataclass, field
from open_rqa.trainers.qa_llm.datasets import SupervisedRQADataset
from open_rqa.trainers.qa_llm.supervised_trainer import SupervisedTrainer
from open_rqa.trainers.qa_llm.arguments import E2EQATrainingArguments
from open_rqa.trainers.utils import (
    remove_optimizer_weights,
    init_logger,
    create_dir_if_not_exists
)
from open_rqa.retrievers.faiss_retriever import FaissRetriever
from open_rqa.evaluation.evaluator import EvaluatorConfig
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import open_rqa.trainers.dist_utils as dist_utils
import torch
import wandb
import random
import logging
import sys
import json
import jsonlines
import pickle
import os


os.environ['TOKENIZERS_PARALLELISM'] = 'true'


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
        default="lmsys/vicuna-7b-v1.5",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention (if supported)"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: str = field(
        default="data/training/databricks_new/train_w_qa.jsonl",
        metadata={"help": "Path for cached train dataset"},
    )
    eval_file: str = field(
        default='data/training/databricks_new/eval_w_qa.jsonl',
        metadata={"help": "Path for cached eval dataset"},
    )
    test_file: str = field(
        default='data/training/databricks_new/test_w_qa.jsonl',
        metadata={"help": "Path for cached test dataset"},
    )
    full_dataset_file_path: str = field(
        default='data/database/databricks/databricks_400.pkl',
        metadata={"help": "Path for cached full dataset file"},
    )
    full_dataset_index_path: str = field(
        default='data/database/databricks/databricks_400_tmp',
        metadata={"help": "Path for cached full dataset index. If first time, this will be created."},
    )
    assistant_prefix: str = field(
        default="ASSISTANT",
        metadata={"help": "Prefix for assistant in a conversation"},
    )
    user_prefix: str = field(
        default="USER",
        metadata={"help": "Prefix for user in a conversation"},
    )
    sep_user: str = field(
        default=" ",
        metadata={"help": "Token right after user finished his/her turn"},
    )
    sep_sys: str = field(
        default="</s>",
        metadata={"help": "Token right after assistant finished his/her turn"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length, INCLUDING source documents"},
    )
    eval_embedding_model: str = field(
        default='text-embedding-ada-002',
        metadata={"help": "The embedding model used for E2E evaluation."},
    )


def init_datasets(data_args: DataArguments, tokenizer):
    with jsonlines.open(data_args.train_file) as fread:
        train_data = list(fread)
    with jsonlines.open(data_args.eval_file) as fread:
        eval_data = list(fread)
    with jsonlines.open(data_args.test_file) as fread:
        test_data = list(fread)
    
    train_dset = SupervisedRQADataset(
        qa_w_doc_data=train_data,
        tokenizer=tokenizer,
        assistant_prefix=data_args.assistant_prefix,
        user_prefix=data_args.user_prefix,
        sep_user=data_args.sep_user,
        sep_sys=data_args.sep_sys,
        max_length=data_args.max_seq_length,
        end_data_idx=None,
        shuffle=True
    )
    eval_dset = SupervisedRQADataset(
        qa_w_doc_data=eval_data,
        tokenizer=tokenizer,
        assistant_prefix=data_args.assistant_prefix,
        user_prefix=data_args.user_prefix,
        sep_user=data_args.sep_user,
        sep_sys=data_args.sep_sys,
        max_length=data_args.max_seq_length,
        end_data_idx=None,
        shuffle=True
    )
    test_dset = SupervisedRQADataset(
        qa_w_doc_data=test_data,
        tokenizer=tokenizer,
        assistant_prefix=data_args.assistant_prefix,
        user_prefix=data_args.user_prefix,
        sep_user=data_args.sep_user,
        sep_sys=data_args.sep_sys,
        max_length=data_args.max_seq_length,
        end_data_idx=None,
        shuffle=True
    )
    return train_dset, eval_dset, test_dset


def init_embedding_model(model_name):
    if model_name in ['text-embedding-ada-002']:
        return OpenAIEmbeddings(
            model=model_name,
            organization=os.environ['OPENAI_ORGANIZATION']
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=model_name
        )


def init_retriever_for_eval(data_args: DataArguments):
    embedding_model = init_embedding_model(data_args.eval_embedding_model)

    with open(data_args.full_dataset_file_path, 'rb') as fread:
        full_dataset = pickle.load(fread)
    logger.info(f"Embedding {len(full_dataset)} documents from {data_args.full_dataset_file_path}")
    
    eval_retriever = FaissRetriever(
        full_dataset,
        embeddings=embedding_model,
        index_path=data_args.full_dataset_index_path
    )
    logger.info("initialized retriever for evaluation")
    return eval_retriever


def main(model_args: ModelArguments, data_args: DataArguments, logger_args: LoggerArguments, training_args: E2EQATrainingArguments):
    random.seed(0)

    logger.info('training with gold (q, a, doc) pairs.')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dset, eval_dset, test_dset = init_datasets(data_args, tokenizer)
    eval_retriever = init_retriever_for_eval(data_args)
    
    if model_args.use_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )

    ### if it is already initialized, huggingface will use it
    all_args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'logger_args': vars(logger_args),
        'training_args': training_args.to_dict()
    }
    if 'wandb' in training_args.report_to:
        _ = wandb.init(
            project='tamarin',
            entity='tamarin',
            name=training_args.output_dir.split("/")[-1] or None,
            group=logger_args.run_group,
            config=all_args,
        )
    
    eval_config = EvaluatorConfig(  # type: ignore
        retr_latency = False,
        gen_f1 = True,
        gen_precision = True,
        gen_rouge = True,
        gen_latency = True,
        e2e_latency = True,
        ## eval model related configs
        assistant_prefix = data_args.assistant_prefix,
        user_prefix = data_args.user_prefix,
        sep_user = data_args.sep_user,
        sep_sys = data_args.sep_sys,
    )

    if training_args.deepspeed is not None:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    training_args.eval_data_path = data_args.eval_file
    trainer = SupervisedTrainer(
        model=model,
        train_args=training_args,
        eval_config=eval_config,
        eval_retriever=eval_retriever,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        tokenizer=tokenizer,
    )
    trainer.train()

    if dist_utils.is_main():
        remove_optimizer_weights(training_args.output_dir)

    # test
    trainer.args.eval_data_path = data_args.test_file
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
    with open(os.path.join(training_args.output_dir, 'all_args.json'), 'w', encoding='utf-8') as f:
        all_args = {
            'model_args': vars(model_args),
            'data_args': vars(data_args),
            'logger_args': vars(logger_args),
            'training_args': training_args.to_dict()
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    
    logger = init_logger(is_main=True, filename=os.path.join(training_args.output_dir, 'train.log'))

    main(model_args, data_args, logger_args, training_args)