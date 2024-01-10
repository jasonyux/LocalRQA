from transformers import HfArgumentParser
from open_rqa.pipelines.retrieval_qa import SimpleRQA
from open_rqa.retrievers.faiss_retriever import FaissRetriever
from open_rqa.evaluation.evaluator import E2EEvaluator, EvaluatorConfig
from open_rqa.trainers.utils import init_logger, create_dir_if_not_exists
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.constants import OPENAI_MODEL_NAMES
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from dataclasses import dataclass, field
from typing import List, Dict
import pickle
import logging
import os
import sys
import json
import jsonlines


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    qa_model_name_or_path: str = field(
        default="lmsys/vicuna-7b-v1.5",
        metadata={"help": "QA model name or path. Huggingface model or OpenAI model"},
    )
    embedding_model_name_or_path: str = field(
        default="intfloat/e5-base",
        metadata={"help": "Embedding model name or path. Huggingface model or OpenAI model"},
    )

@dataclass
class TestArguments:
    document_path: str = field(
        default='data/training/databricks_sources_official_short.pkl',
        metadata={"help": "Path to the file which contains List[Document] for building a database index"},
    )
    index_path: str = field(
        default='data/training/databricks_sources_official_short_index',
        metadata={"help": "Path to the file which will store/contains the index of documents in document_path"},
    )
    eval_data_path: str = field(
        default='data/training/databricks_clean/test_q_doc_a.jsonl',
        metadata={
            "help": ("Path to the eval data JSONL file. It needs to contain fields including 'gold_docs' for retriever, "
                    "and 'gold_docs' and 'gold_answers' for E2E QA.")
        },
    )
    output_dir: str = field(
        default="model_checkpoints/debug_test",
        metadata={"help": "Path to the output directory for saving test predictions and results"},
    )


def init_rqa_model(model_args: ModelArguments, documents: List[Document], index_path: str):
    ### init retriever
    if model_args.embedding_model_name_or_path in OPENAI_MODEL_NAMES:
        embedding_model = OpenAIEmbeddings(
            model=model_args.embedding_model_name_or_path,
            organization=os.environ['OPENAI_ORGANIZATION']
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_args.embedding_model_name_or_path
        )
    logger.info(f"Initializing retriever with {model_args.embedding_model_name_or_path} and {index_path}")
    retriever = FaissRetriever(
        documents,
        embeddings=embedding_model,
        index_path=index_path
    )

    ### init qa model
    logger.info(f"Initializing qa model with {model_args.qa_model_name_or_path}")
    if model_args.qa_model_name_or_path in OPENAI_MODEL_NAMES:
        raise NotImplementedError
    else:
        rqa_model = SimpleRQA.from_huggingface(
            retriever=retriever,
            qa_model_name_or_path=model_args.qa_model_name_or_path,
            user_prefix="USER",  # doesn't really matter as evaluation during training is single turn
            assistant_prefix="ASSISTANT",
        )
    return rqa_model


def load_eval_data(eval_data_path) -> List[Dict]:
    with jsonlines.open(eval_data_path) as fread:
        eval_data = list(fread)
    formatted_eval_data = []
    for d in eval_data:
        gold_doc = Document.from_dict(d['gold_doc'])
        formatted_eval_data.append({
            'question': d['question'],
            'gold_doc': gold_doc,
            'gold_docs': [gold_doc],  # compatibiliy with E2EEvaluator
            'gold_answer': d['gold_answer'],
            'dialogue_session': DialogueSession.from_list(d['chat_history']),
        })
    return formatted_eval_data


def test(model_args: ModelArguments, test_args: TestArguments):
    ### load documents database
    with open(test_args.document_path, 'rb') as fread:
        documents = pickle.load(fread)
    logger.info(f"Loaded {len(documents)} documents from {test_args.document_path}")

    ### init rqa model
    rqa_model = init_rqa_model(model_args, documents, test_args.index_path)

    ### evaluation
    eval_config = EvaluatorConfig(  # type: ignore
        retr_latency = False,
        gen_f1 = True,
        gen_precision = True,
        gen_rouge = True,
        gen_latency = True,
        e2e_latency = True,
    )
    loaded_eval_data = load_eval_data(test_args.eval_data_path)
    evaluator = E2EEvaluator(
        config=eval_config,
        test_data=loaded_eval_data,
    )
    performance, predictions = evaluator.evaluate(rqa_model, prefix='test')
    logger.info(f"Performance: {json.dumps(performance, indent=2, sort_keys=True)}")

    ### write prections
    save_path = os.path.join(test_args.output_dir, 'test-predictions.pkl')
    with open(save_path, 'wb') as fwrite:
        pickle.dump(predictions, fwrite)
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