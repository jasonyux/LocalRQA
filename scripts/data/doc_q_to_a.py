from open_rqa.qa_llms.base import BaseQAModel
from open_rqa.qa_llms.openai import OpenAIQAModel
from open_rqa.qa_llms.huggingface import HuggingFaceQAModel
from open_rqa.constants import OPENAI_MODEL_NAMES
from open_rqa.trainers.utils import init_logger
from typing import Dict, List
from tqdm.auto import tqdm
from copy import deepcopy
import pickle
import logging
import random
import jsonlines
import os
import math
import argparse


logger = logging.getLogger(__name__)


BASE_DOCQ2A_PROMPT = """
The following texts are extracted from a company's documentations. Your task is to answer user's questions based on the following documents.
----
{fmt_content}
----
Answer the following question using the document provided above.
Question: {question}
Answer:
""".strip()


def add_parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--prompt_model", type=str, default="gpt-3.5-turbo",
        help=("Model to prompt for getting questions from documents. Can be either openai models or huggingface models. "
        "For huggingface models, make sure you adjust the batch size to accomodate for GPU memory usage.")
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for question generation. If we are using vicuna than about 2 takes 70+GB of GPU memory."
    )
    parser.add_argument(
        "--dataset_w_q", type=str, default="data/training/databricks_new/test.jsonl",
        help="Path to the pickle file that stores List[Dict] representing the dataset with questions. Fields include questions, gold_docs, etc.",
    )
    parser.add_argument(
        "--end_data_idx", type=int, default=None,
        help="End index of the data to process"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Path to save ALL the generated data"
    )
    parser.add_argument(
        "--save_name", type=str, default="test_w_a.jsonl",
        help="Name of the save file"
    )
    return parser


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    return args


def load_rqa_data(args: argparse.Namespace) -> List[Dict]:
    with jsonlines.open(args.dataset_w_q) as fread:
        to_gen_data = list(fread)

    flattened_data = []
    for d in to_gen_data:
        assert 'questions' in d or 'question' in d, f'Unknown data format: {d.keys()}'
        if 'questions' in d:
            # need to flatten
            for q in d['questions']:
                new_d = deepcopy(d)
                new_d.pop('questions', None)
                new_d['question'] = q
                flattened_data.append(new_d)
        else:
            # already flat
            d.pop('questions', None)
            flattened_data.append(d)
    
    logger.info(f"Found number of sources: {len(flattened_data)}")
    return flattened_data


def init_prompting_model(args: argparse.Namespace):
    prompting_model: BaseQAModel

    if args.prompt_model in OPENAI_MODEL_NAMES:
        prompting_model = OpenAIQAModel(
            model_name = args.prompt_model
        )
    else:
        # assume it is huggingface
        prompting_model = HuggingFaceQAModel(
            model_name_or_path = args.prompt_model
        )
    return prompting_model


def _list_to_iterator(list_data: list, batch_size = 8):
    batch = []
    for item in list_data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def _batch_generate_answers(prompting_model: BaseQAModel, batch: List[Dict], prompt: str):
    ## prepare prompts
    all_input_texts = []
    for sample in batch:
        # format all docs
        gold_docs_fmted = "\n".join([d['fmt_content'] for d in sample['gold_docs']])
        question = sample['question']
        input_text = prompt.format(
            fmt_content = gold_docs_fmted,
            question = question,
        )
        all_input_texts.append(input_text)

    ## generate answers
    if isinstance(prompting_model, HuggingFaceQAModel):
        tokenize_kwargs = {
            'padding': 'longest',
            'return_tensors': "pt",
        }
        generate_kwargs = {
            "max_new_tokens": 256,
            "num_beams": 1,
        }
    elif isinstance(prompting_model, OpenAIQAModel):
        tokenize_kwargs = {}
        generate_kwargs = {
            "max_tokens": 256,
        }
    else:
        raise ValueError(f"Unknown prompting model type: {type(prompting_model)}")

    output = prompting_model.generate(
        batched_prompts=all_input_texts,
        tokenization_kwargs=tokenize_kwargs,
        generation_kwargs=generate_kwargs,
    ).batch_answers

    cleaned_answers = []
    for o in output:
        cleaned_answers.append(o.strip())
    return cleaned_answers


def _generate_questions_from_dataset(args, prompting_model: BaseQAModel, doc_w_q_dataset: List[Dict], prompt):
    dataset = []
    num_steps = math.ceil(len(doc_w_q_dataset) / args.batch_size)
    iterator = _list_to_iterator(doc_w_q_dataset, batch_size=args.batch_size)

    pbar = tqdm(total=num_steps, desc="Generating answers")
    for i, batch in enumerate(iterator):
        answers = _batch_generate_answers(
            prompting_model = prompting_model,
            batch = batch,
            prompt = prompt,
        )
        
        for sample, answer in zip(batch, answers):
            sample['gold_answer'] = answer
            dataset.append(sample)

        pbar.update(1)
        if i % 10 == 0 and i > 0:
            logger.info(f'Saving dataset at step {i}')
            with open(os.path.join(args.save_dir, "_raw_a.pkl"), "wb") as f:
                pickle.dump(dataset, f)
    return dataset

def generate_questions_from_dataset(args, prompt_template):
    to_gen_data = load_rqa_data(args)

    if args.end_data_idx is not None:
        random.shuffle(to_gen_data)
        to_gen_data = to_gen_data[:args.end_data_idx]  # if you want to subsample
    
    logger.info(f"Generating answers for {len(to_gen_data)} (doc, q) pairs")
    prompting_model = init_prompting_model(args)

    generated_data = _generate_questions_from_dataset(
        args,
        prompting_model = prompting_model,
        doc_w_q_dataset = to_gen_data,
        prompt = prompt_template,
    )
    return generated_data


def main(args: argparse.Namespace):
    random.seed(0)
    data_w_questions = generate_questions_from_dataset(
        args,
        prompt_template = BASE_DOCQ2A_PROMPT,  # customizable
    )
    
    logger.info(f"Generated {len(data_w_questions)} data with answers")
    logger.info(f"Saving to {os.path.join(args.save_dir, args.save_name)}")
    with jsonlines.open(os.path.join(args.save_dir, args.save_name), "w") as fwrite:
        fwrite.write_all(data_w_questions)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate (document, question, answer) pairs given a (document, question) pairs. "
        "NOTE: for this script to work properly, we assume data being a list of Dict having keys [gold_docs, questions]" )
    )
    parser = add_parser_arguments(parser)
    args = parse_arguments(parser)

    logger = init_logger(filename=None)

    main(args)