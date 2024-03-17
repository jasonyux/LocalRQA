from transformers import HfArgumentParser
from local_rqa.pipelines.retrieval_qa import SimpleRQA
from local_rqa.evaluation.evaluator import E2EEvaluator, EvaluatorConfig
from local_rqa.trainers.utils import init_logger, create_dir_if_not_exists
from local_rqa.schema.document import Document
from local_rqa.schema.dialogue import DialogueSession
from dataclasses import dataclass, field
from typing import List, Dict
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
    qa_is_fid: bool = field(
        default=False,
        metadata={"help": "Whether the QA model is a FiD model"},
    )
    embedding_model_name_or_path: str = field(
        default="intfloat/e5-base",
        metadata={"help": "Embedding model name or path. Huggingface model or OpenAI model"},
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


@dataclass
class TestArguments:
    document_path: str = field(
        default='example/databricks/database/databricks.pkl',
        metadata={"help": "Path to the file which contains List[Document] for building a database index"},
    )
    index_path: str = field(
        default='data/database/databricks/databricks_400_e5-base',
        metadata={"help": "Path to the file which will store/contains the index of documents in document_path"},
    )
    eval_data_path: str = field(
        default='example/databricks/processed/test_w_qa.jsonl',
        metadata={
            "help": ("Path to the eval data JSONL file. It needs to contain fields including 'gold_docs' for retriever, "
                    "and 'gold_docs' and 'gold_answers' for E2E QA.")
        },
    )
    output_dir: str = field(
        default="model_checkpoints/debug_test",
        metadata={"help": "Path to the output directory for saving test predictions and results"},
    )
    ### eval config
    gen_gpt4eval: bool = field(
        default=False,
        metadata={"help": "Whether to use GPT4 for evaluation"},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for evaluation"},
    )


def init_rqa_model(model_args: ModelArguments, document_path: str, index_path: str):
    ### init retriever
    rqa_model = SimpleRQA.from_scratch(
        document_path=document_path,
        index_path=index_path,
        qa_model_name_or_path=model_args.qa_model_name_or_path,
        qa_is_fid=model_args.qa_is_fid,
        embedding_model_name_or_path=model_args.embedding_model_name_or_path,
        assistant_prefix=model_args.assistant_prefix,
        user_prefix=model_args.user_prefix,
        sep_user=model_args.sep_user,
        sep_sys=model_args.sep_sys,
    )
    return rqa_model


def load_eval_data(eval_data_path) -> List[Dict]:
    with jsonlines.open(eval_data_path) as fread:
        eval_data = list(fread)
    formatted_eval_data = []
    for d in eval_data:
        formatted_eval_data.append({
            'question': d['question'],
            'gold_docs': [Document.from_dict(doc) for doc in d['gold_docs']],
            'gold_answer': d['gold_answer'],
            'dialogue_session': DialogueSession.from_list(d['chat_history']),
        })
    return formatted_eval_data


def test(model_args: ModelArguments, test_args: TestArguments):
    ### init rqa model
    rqa_model = init_rqa_model(model_args, test_args.document_path, test_args.index_path)

    ### evaluation
    eval_config = EvaluatorConfig(  # type: ignore
        batch_size = test_args.batch_size,
        retr_latency = False,
        gen_f1 = True,
        gen_precision = True,
        gen_rouge = True,
        gen_latency = True,
        gen_gpt4eval = test_args.gen_gpt4eval,
        e2e_latency = True,
        ## eval model related configs
        assistant_prefix = model_args.assistant_prefix,
        user_prefix = model_args.user_prefix,
        sep_user = model_args.sep_user,
        sep_sys = model_args.sep_sys,
    )
    loaded_eval_data = load_eval_data(test_args.eval_data_path)
    evaluator = E2EEvaluator(
        config=eval_config,
        test_data=loaded_eval_data,
    )
    performance, predictions = evaluator.evaluate(rqa_model, prefix='test')
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