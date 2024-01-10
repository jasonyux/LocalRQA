from open_rqa.qa_llms.openai import OpenAIQAModel
from open_rqa.qa_llms.huggingface import HuggingFaceQAModel
from open_rqa.qa_llms.base import BaseQAModel
from open_rqa.trainers.utils import init_logger
from open_rqa.schema.document import Document
from open_rqa.evaluation.utils import normalize_answer
from open_rqa.evaluation.metrics import is_almost_same_document
from open_rqa.constants import OPENAI_MODEL_NAMES
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Dict, List, Callable
from multiprocessing import Pool
import argparse
import os
import pickle
import re
import logging
import random
import math
import evaluate


logger = logging.getLogger(__name__)


BASE_DOC2Q_PROMPT = """
The following texts are extracted from a company's documentations. Your task is to create questions that users might ask if they have not read the documentations.
------
{fmt_content}
------
Create two questions that a user might ask if they have not read these texts. Only create questions that can be answered using the texts above.
Question 1:
""".strip()


def add_parser_arguments(parser):
    parser.add_argument(
        "-mode", type=str, required=True,
        choices = ["all", "init_eval_dset", "create_eval_dset", "create_train_dset"],
        help=("Mode to run. Ideally the order would be (manually): init_eval_dset + manually check eval sets -> create_eval_dset + manually check eval sets -> create_train_dset. "
        "The quick version to just run everything would be: all = init_eval_dset + create_eval_dset + create_train_dset")
    )
    parser.add_argument(
        "-document_path", type=str, required=True,
        help="Path to the chunked documents, i.e., the pickle file used as database for retriever"
    )
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
        "--num_hard_negs_per_doc", type=int, default=2,
        help="Number of hard negative examples per gold document"
    )
    parser.add_argument(
        "--num_train_data", type=int, default=1000,
        help="Number of data to generate for training. Specify -1 to use all (non eval or test) data"
    )
    parser.add_argument(
        "--num_eval_test_data", type=int, default=200,
        help="Number of data to generate for eval and test."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Path to save ALL the generated data"
    )
    return parser


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    if args.num_train_data == -1:
        args.num_train_data = None
    return args


def load_documents(args: argparse.Namespace) -> Dict[str, Document]:
    with open(args.document_path, "rb") as fread:
        texts = pickle.load(fread)

    source_to_doc = defaultdict(list)
    for doc in texts:
        # used to re-construct which document a document chunk belongs to
        source_to_doc[doc.metadata['source']].append(doc)

    logger.info(f"Found number of sources: {len(source_to_doc)}")
    return source_to_doc


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


def __get_unique_documents(data):
    source, documents = data  # optimized using multiprocessing
    unique_documents = []
    last_k = 3
    # do it in a single pass
    for doc in documents:
        doc: Document
        if len(normalize_answer(doc.page_content).split()) == 0:
            # empty document
            continue

        is_unique_doc = True
        last_k_docs = unique_documents[-last_k:]
        for last_doc in last_k_docs:
            if is_almost_same_document(doc, last_doc, threshold=0.5):
                is_unique_doc = False
                break
        
        if is_unique_doc:
            unique_documents.append(doc)
    return source, unique_documents


def _get_unique_documents(source_to_doc: Dict[str, Document]):
    unique_documents_per_source = {}
    # for source, documents in tqdm(source_to_doc.items(), desc="Getting unique documents per source"):
    #     unique_documents_per_source[source] = __get_unique_documents(documents)
    pbar = tqdm(total=len(source_to_doc), desc="Getting unique documents per source")
    with Pool(processes=8) as pool:
        all_args = []
        for s, d in source_to_doc.items():
            all_args.append((s, d))
    
        for source, unique_documents in pool.imap_unordered(__get_unique_documents, all_args):
            unique_documents_per_source[source] = unique_documents
            pbar.update(1)
    return unique_documents_per_source


def create_positive_n_negative_examples(args: argparse.Namespace, filter_fn: Callable):
    source_to_doc: Dict[str, Document] = load_documents(args)
    num_hard_negs_per_doc = args.num_hard_negs_per_doc

    documents_dataset = []
    # hard negatives are document chunks from the SAME source
    # but first we remove document chunks that have a high overlap
    source_to_unique_documents = _get_unique_documents(source_to_doc)
    for source in tqdm(source_to_unique_documents, desc="Constructing positive and hard negative documents for qa"):
        # first we consider all possible positive pairs
        unique_docs = source_to_unique_documents[source]
        if len(unique_docs) < num_hard_negs_per_doc + 1:
            continue

        # pairs where we CAN create gold and hard negatives
        # further more, we can create pairs for EVERY document in unique_documents_per_source
        for j, gold_doc in enumerate(unique_docs):
            if not filter_fn(gold_doc):
                continue

            # sample hard negative
            hard_neg_docs = random.sample(unique_docs[:j] + unique_docs[j+1:], num_hard_negs_per_doc)

            # create examples
            sample = {
                "gold_doc": gold_doc,
                "hard_neg_docs": hard_neg_docs,
                "chat_history_str": "",  # one turn QA. Used by our evaluation interface
            }
            documents_dataset.append(sample)
    
    ### save
    save_path = os.path.join(args.save_dir, "all_doc_neg_pairs.pkl")
    logger.info(f"Saving to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(documents_dataset, f)
    return documents_dataset


def _extract_questions(generated_output):
    lines = generated_output.split("\n")
    if not lines[0].strip().endswith("?"):
        return []
    gen_questions = set([lines[0].strip()])
    for line in lines[1:3]:
        if re.search(r"Question [2-3]:(.+)", line) is not None:
            extracted_q = re.search(r"Question [2-3]:(.+)", line).group(1).strip()
            gen_questions.add(extracted_q)
    return list(gen_questions)


def _list_to_iterator(list_data: list, batch_size = 8):
    batch = []
    for item in list_data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def _batch_generate_questions(prompting_model: BaseQAModel, documents: List[Document], prompt, verbose=False):
    ## prepare prompts
    all_input_texts = []
    for doc in documents:
        input_text = prompt.format(fmt_content=doc.fmt_content)
        all_input_texts.append(input_text)

    ## generate questions
    if isinstance(prompting_model, HuggingFaceQAModel):
        tokenize_kwargs = {
            'padding': 'longest',
            'return_tensors': "pt",
        }
        generate_kwargs = {
            "max_new_tokens": 256,
            "num_beams": 1,
            "early_stopping": True,
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

    extracted_questions = []
    for o in output:
        # extract the question
        questions = _extract_questions(o)
        if verbose:
            logger.info(f"Extracted questions: \n{questions}")
        extracted_questions.append(questions)
    return extracted_questions


def _batch_generate_questions_from_dataset(prompting_model: BaseQAModel, documents_dataset, prompt, batch_size=8, num_data=1000):
    dataset = []
    num_steps = math.ceil(len(documents_dataset) / batch_size) if num_data is None else math.ceil(num_data / batch_size)
    iterator = _list_to_iterator(documents_dataset, batch_size=batch_size)

    pbar = tqdm(total=num_steps, desc="Generating questions")
    for i, batched_samples in enumerate(iterator):
        questions = _batch_generate_questions(
            prompting_model,
            [s['gold_doc'] for s in batched_samples],
            prompt = prompt,
        )

        # collect the new data
        for sample, q in zip(batched_samples, questions):
            if len(q) == 0:
                continue
            
            new_sample = {
                **sample,
                'questions': q,
            }
            dataset.append(new_sample)
        
            if num_data is not None and len(dataset) >= num_data:
                break
        pbar.update(1)
        if num_data is not None and len(dataset) >= num_data:
            break

        if i % 10 == 0 and i > 0:
            logger.info(f'Saving intermediate dataset at step {i}')
            with open(os.path.join(args.save_dir, "_raw.pkl"), "wb") as f:
                pickle.dump(dataset, f)
    return dataset


def create_heldout_test_dset(args, doc2q_prompt):
    prompting_model = init_prompting_model(args)

    all_doc_neg_pairs_path = os.path.join(args.save_dir, "all_doc_neg_pairs.pkl")
    with open(all_doc_neg_pairs_path, "rb") as fread:
        all_documents_dataset = pickle.load(fread)
    logger.info(f"Loaded all q2docs pairs: {len(all_documents_dataset)}")
    
    # create pairs for eval or test
    random.shuffle(all_documents_dataset)
    non_train_dset = _batch_generate_questions_from_dataset(
        prompting_model, all_documents_dataset,
        prompt=doc2q_prompt,
        batch_size=args.batch_size,
        num_data=args.num_eval_test_data
    )

    half_size = len(non_train_dset) // 2
    eval_dataset = non_train_dset[:half_size]
    test_dataset = non_train_dset[half_size:]

    ### save
    logger.info(f"Saving to {args.save_dir} named eval.pkl, test.pkl, nontrain.pkl")
    with open(os.path.join(args.save_dir, "nontrain.pkl"), "wb") as f:
        pickle.dump(non_train_dset, f)
    with open(os.path.join(args.save_dir, "eval.pkl"), "wb") as f:
        pickle.dump(eval_dataset, f)
    with open(os.path.join(args.save_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_dataset, f)
    
    ### since evaluation need to reembed the entire dataset, we consider a smaller document set for an optional fast evaluation
    gather_n_save_documents(args, eval_dataset, "eval_documents.pkl")
    gather_n_save_documents(args, test_dataset, "test_documents.pkl")
    return eval_dataset, test_dataset


def gather_n_save_documents(args, eval_dataset, save_name: str):
    seen_docs = {}

    all_docs_used = []
    for d in eval_dataset:
        all_curr_docs = [d['gold_doc']]
        all_curr_docs.extend(d['hard_neg_docs'])

        for doc in all_curr_docs:
            url = doc.metadata['source']
            content = doc.page_content
            if url in seen_docs and content in seen_docs[url]:
                continue
            seen_docs[url] = seen_docs.get(url, set())
            seen_docs[url].add(content)

            all_docs_used.append(doc)
    
    logger.info(f"Saving gather_n_save_documents to {args.save_dir} named {save_name}")
    with open(os.path.join(args.save_dir, save_name), "wb") as f:
        pickle.dump(all_docs_used, f)
    return


def _hash_document(doc: Document):
    return doc.fmt_content


def _normalize_question(q: str):
    q = q.lower().strip()
    return q


def create_train_dset(args, doc2q_prompt):
    prompting_model = init_prompting_model(args)
    
    with open(os.path.join(args.save_dir, "all_doc_neg_pairs.pkl"), "rb") as fread:
        documents_dataset = pickle.load(fread)
    logger.info(f"Loaded all (doc, neg_docs) pairs: {len(documents_dataset)}")
    with open(os.path.join(args.save_dir, "nontrain.pkl"), "rb") as fread:
        held_out_documents_dataset = pickle.load(fread)
    logger.info(f"Loaded held out q2docs pairs: {len(held_out_documents_dataset)}")

    ### make sure eval and test test are not in the training set: either same document but NEW questions, or new documents
    eval_n_test_doc2q_mapping: Dict[str, set] = {}
    for sample in held_out_documents_dataset:
        gold_doc = sample['gold_doc']
        doc_id = _hash_document(gold_doc)
        if doc_id not in eval_n_test_doc2q_mapping:
            eval_n_test_doc2q_mapping[doc_id] = set()

        normalized_questions = [_normalize_question(q) for q in sample['questions']]
        eval_n_test_doc2q_mapping[doc_id].update(normalized_questions)

    random.shuffle(documents_dataset)
    potential_train_dataset = _batch_generate_questions_from_dataset(
        prompting_model, documents_dataset, 
        prompt=doc2q_prompt,
        batch_size=args.batch_size,
        num_data=args.num_train_data
    )

    ### make sure eval and test set are not in the training set
    rouge = evaluate.load('rouge')
    train_dataset = []
    removed_duplicates = 0
    for sample in potential_train_dataset:
        gold_doc = sample['gold_doc']
        doc_id = _hash_document(gold_doc)
        if doc_id not in eval_n_test_doc2q_mapping:
            train_dataset.append(sample)
        else:
            # use rouge to make sure the questions are not the same
            original_questions = sample['questions']
            normalized_questions = [_normalize_question(q) for q in original_questions]
            new_questions = []

            for i, q in enumerate(normalized_questions):
                # dont include questions if rougeL > 0.5, otherwise we might "see the eval/test set"
                rg_scores = [rouge.compute(predictions=[q], references=[qq])['rougeL'] for qq in eval_n_test_doc2q_mapping[doc_id]]
                if all(sc < 0.5 for sc in rg_scores):
                    new_questions.append(original_questions[i])

            ### add only if these are new questions, given that the document is the same
            if len(new_questions) > 0:
                sample['questions'] = new_questions
                train_dataset.append(sample)
            else:
                removed_duplicates += 1
    
    logger.info(f"Removed {removed_duplicates} duplicates")

    ## save
    logger.info(f"Saving to {args.save_dir} named train.pkl")
    with open(os.path.join(args.save_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)
    return train_dataset


def main(args: argparse.Namespace):
    """to customize how (doc, q) pairs would be created, simply copy this function over and modify the "# customizable" parts
    """
    random.seed(0)
    if args.mode in ["init_eval_dset", "all"]:
        documents_dataset = create_positive_n_negative_examples(
            args=args,
            filter_fn=lambda doc: True  # customizable
        )
        logger.info(f"Created {len(documents_dataset)} <gold document, hard negative documents> pairs.")
    if args.mode in ["create_eval_dset", "all"]:
        eval_dataset, test_dataset = create_heldout_test_dset(
            args,
            doc2q_prompt=BASE_DOC2Q_PROMPT  # customizable
        )
        logger.info(f"Number of eval samples: {len(eval_dataset)}")
        logger.info(f"Number of test samples: {len(test_dataset)}")
    if args.mode in ["create_train_dset", "all"]:
        train_dataset = create_train_dset(
            args,
            doc2q_prompt=BASE_DOC2Q_PROMPT  # customizable
        )
        logger.info(f"Number of train samples: {len(train_dataset)}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate (document, question) pairs given a (chunked) document database. This can be used for generating both testing (q, doc) pairs AND training (q, doc) pairs. "
        "NOTE: for this script to work properly, we assume document.metadata['source'] is NOT EMPTY (e.g., can be the url of the unchunked document, the first level title, etc.)" )
    )
    parser = add_parser_arguments(parser)
    args = parse_arguments(parser)
    
    logger = init_logger(filename=None)

    main(args)