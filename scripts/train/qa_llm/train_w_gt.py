from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    HfArgumentParser
)
from accelerate.utils import DistributedType
from dataclasses import dataclass, field
from open_rqa.trainers.qa_llm.datasets import SupervisedRQAwRetrieverDataset
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
from functools import partial
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
        default="data/training/databricks_clean/train_q_doc_a.jsonl",
        metadata={"help": "Path for cached train dataset"},
    )
    eval_file: str = field(
        default='data/training/databricks_clean/eval_q_doc_a.jsonl',
        metadata={"help": "Path for cached eval dataset"},
    )
    test_file: str = field(
        default='data/training/databricks_clean/test_q_doc_a.jsonl',
        metadata={"help": "Path for cached test dataset"},
    )
    full_dataset_file_path: str = field(
        default='data/training/databricks_sources_official_short.pkl',
        metadata={"help": "Path for cached full dataset file"},
    )
    full_dataset_index_path: str = field(
        default='data/training/databricks_sources_official_short_index',
        metadata={"help": "Path for cached full dataset index"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length, INCLUDING source documents"},
    )
    embedding_model: str = field(
        default="",
        metadata={"help": "What embedding model to train with (e.g., intfloat/e5-base). If empty, train with ground truth."},
    )
    embedding_max_num_to_retrieve: int = field(
        default=2,
        metadata={"help": "Max number of documents to retrieve (excluding the gold doc), if embedding_model is none empty"},
    )



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


def retriever_init_fn(embedding_model, documents, index_path):
    retriever = FaissRetriever(
        documents,
        embeddings=embedding_model,
        index_path=index_path
    )
    return retriever


def init_datasets(data_args: DataArguments, tokenizer, tmp_output_dir: str, embedding_model):
    with jsonlines.open(data_args.train_file) as fread:
        train_data = list(fread)
    with jsonlines.open(data_args.eval_file) as fread:
        eval_data = list(fread)
    with jsonlines.open(data_args.test_file) as fread:
        test_data = list(fread)
    
    train_index_save_path = os.path.join(tmp_output_dir, 'train_index')
    train_dset = SupervisedRQAwRetrieverDataset(
        qa_w_doc_data=train_data,
        embedding_model=embedding_model,
        retriever_init_fn=partial(retriever_init_fn, index_path=train_index_save_path),
        max_num_to_retrieve=data_args.embedding_max_num_to_retrieve,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        end_data_idx=200,
        shuffle=True
    )
    eval_index_save_path = os.path.join(tmp_output_dir, 'eval_index')
    eval_dset = SupervisedRQAwRetrieverDataset(
        qa_w_doc_data=eval_data,
        embedding_model=embedding_model,
        retriever_init_fn=partial(retriever_init_fn, index_path=eval_index_save_path),
        max_num_to_retrieve=data_args.embedding_max_num_to_retrieve,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        end_data_idx=None,
        shuffle=False
    )
    test_index_save_path = os.path.join(tmp_output_dir, 'test_index')
    test_dset = SupervisedRQAwRetrieverDataset(
        qa_w_doc_data=test_data,
        embedding_model=embedding_model,
        retriever_init_fn=partial(retriever_init_fn, index_path=test_index_save_path),
        max_num_to_retrieve=data_args.embedding_max_num_to_retrieve,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        end_data_idx=None,
        shuffle=False
    )
    return train_dset, eval_dset, test_dset


def init_retriever_for_eval(data_args: DataArguments, embedding_model):
    with open(data_args.full_dataset_file_path, 'rb') as fread:
        full_dataset = pickle.load(fread)
    logger.info(f"Embedding {len(full_dataset)} documents from {data_args.full_dataset_file_path}")
    
    eval_retriever = retriever_init_fn(
        embedding_model=embedding_model,
        documents=full_dataset,
        index_path=data_args.full_dataset_index_path
    )
    logger.info("initialized retriever for evaluation")
    return eval_retriever


def main(model_args: ModelArguments, data_args: DataArguments, logger_args: LoggerArguments, training_args: E2EQATrainingArguments):
    random.seed(0)

    logger.info('training with retrieved documents from embedding model')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedding_model = init_embedding_model(data_args.embedding_model)
    train_dset, eval_dset, test_dset = init_datasets(data_args, tokenizer, training_args.output_dir, embedding_model)
    eval_retriever = init_retriever_for_eval(data_args, embedding_model)
    
    if model_args.use_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
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
        gen_f1 = True,
        gen_precision = True,
        gen_rouge = True,
        gen_latency = True,
    )

    if training_args.deepspeed is not None:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
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