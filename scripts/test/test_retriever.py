from transformers import HfArgumentParser
from local_rqa.retrievers.faiss_retriever import FaissRetriever
from local_rqa.retrievers.bm25_retriever import BM25Retriever
from local_rqa.evaluation.evaluator import RetrieverEvaluator, EvaluatorConfig
from local_rqa.trainers.utils import init_logger, create_dir_if_not_exists
from local_rqa.schema.document import Document
from local_rqa.constants import OPENAI_MODEL_NAMES
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from dataclasses import dataclass, field
from typing import List, Dict
import pickle
import logging
import os
import sys
import json
import copy
import jsonlines


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    embedding_model_name_or_path: str = field(
        default="intfloat/e5-base",
        metadata={"help": "Embedding model name or path. Huggingface model or OpenAI model"},
    )


@dataclass
class TestArguments:
    document_path: str = field(
        default='data/database/databricks/databricks_400.pkl',
        metadata={"help": "Path to the file which contains List[Document] for building a database index"},
    )
    index_path: str = field(
        default='data/database/databricks/databricks_400_e5-base',
        metadata={"help": "Path to the file which will store/contains the index of documents in document_path"},
    )
    eval_data_path: str = field(
        default='data/training/databricks_new/test_w_q.jsonl',
        metadata={
            "help": ("Path to the eval data JSONL file. It needs to contain fields including 'gold_docs' for retriever, "
                    "and 'gold_docs' and 'gold_answers' for E2E QA.")
        },
    )
    retirever_type: str = field(
        default="faiss",
        metadata={"help": "faiss or BM25"},
    )
    test_bszv: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    output_dir: str = field(
        default="model_checkpoints/debug_test",
        metadata={"help": "Path to the output directory for saving test predictions and results"},
    )


def init_retriever_model(model_args: ModelArguments, test_args: TestArguments, documents: List[Document]):
    if test_args.retirever_type == "faiss":
        if model_args.embedding_model_name_or_path in OPENAI_MODEL_NAMES:
            embedding_model = OpenAIEmbeddings(
                model=model_args.embedding_model_name_or_path,
                organization=os.environ['OPENAI_ORGANIZATION']
            )
        else:
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_args.embedding_model_name_or_path
            )
        logger.info(f"Initializing retriever with {model_args.embedding_model_name_or_path} and {test_args.index_path}")
        retriever = FaissRetriever(
            documents,
            embeddings=embedding_model,
            index_path=test_args.index_path
        )
    elif test_args.retirever_type == "BM25":
        retriever = BM25Retriever(documents)
    else:
        raise NotImplementedError("Please choose retriever type from [faiss, BM25]")
    return retriever


def load_eval_data(eval_data_path) -> List[Dict]:
    with jsonlines.open(eval_data_path) as fread:
        eval_data = list(fread)
    flattened_eval_data = []
    for d in eval_data:
        for q in d['questions']:
            new_data = copy.deepcopy(d)
            new_data['question'] = q
            flattened_eval_data.append(new_data)
    return flattened_eval_data


def test(model_args: ModelArguments, test_args: TestArguments):
    ### load documents database
    with open(test_args.document_path, 'rb') as fread:
        documents = pickle.load(fread)
    logger.info(f"Loaded {len(documents)} documents from {test_args.document_path}")

    ### init retriever model
    retriever_model = init_retriever_model(model_args, test_args, documents)

    ### evaluation
    eval_config = EvaluatorConfig(
        gen_latency = False,
        batch_size = test_args.test_bszv
    )
    loaded_eval_data = load_eval_data(test_args.eval_data_path)
    evaluator = RetrieverEvaluator(
        config=eval_config,
        test_data=loaded_eval_data,
    )
    performance, predictions = evaluator.evaluate(retriever_model, prefix='test')
    logger.info(f"Performance: {json.dumps(performance, indent=2, sort_keys=True)}")

    ### write prections
    save_path = os.path.join(test_args.output_dir, 'test-predictions.jsonl')
    with jsonlines.open(save_path, 'w') as fwrite:
        fwrite.write_all(predictions)
    # also save performance
    with open(os.path.join(test_args.output_dir, 'score.json'), 'w', encoding='utf-8') as fwrite:
        json.dump(performance, fwrite, indent=2, sort_keys=True)
    return


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ModelArguments, TestArguments),
        description="E2E QA training script"
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, test_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )  # pylint: disable=global-statement
    else:
        # pylint: disable-next=global-statement
        model_args, test_args = parser.parse_args_into_dataclasses()
    print('received model_args:')
    print(json.dumps(vars(model_args), indent=2, sort_keys=True))
    print('received test_args:')
    print(json.dumps(vars(test_args), indent=2, sort_keys=True))
    
    # save config to model_args.model_save_path
    create_dir_if_not_exists(test_args.output_dir)
    with open(os.path.join(test_args.output_dir, 'all_args.json'), 'w', encoding='utf-8') as f:
        all_args = {
            'model_args': vars(model_args),
            'test_args': vars(test_args),
        }
        json.dump(all_args, f, indent=2, sort_keys=True)
    
    
    logger = init_logger(is_main=True, filename=os.path.join(test_args.output_dir, 'test.log'))

    test(model_args, test_args)