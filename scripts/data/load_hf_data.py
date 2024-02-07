from local_rqa.schema.document import Document
from tqdm.auto import tqdm
from datasets import load_dataset
import jsonlines
import pickle
import os


def _process_trivia_qa(example: dict):
    question = example['question']
    ### process documents
    ## all search context are good results
    ranks = example['search_results']['rank']
    gold_docs = []

    for i, _ in enumerate(ranks):
        desc = example['search_results']['description'][i]
        title = example['search_results']['title'][i]
        url = example['search_results']['url'][i]
        context = example['search_results']['search_context'][i]
        doc_id = example['search_results']['filename'][i]

        file_content = f"""
        title: {title}
        description: {desc}
        context: {context}
        """.replace("    ","").strip()

        doc = Document(
            page_content=file_content,
            metadata={
                "url": url,
                "source": url,
                "doc_id": doc_id
            }
        )
        gold_docs.append(doc)

    return {
        'chat_history': [],
        'hard_neg_docs': [],
        'question': question,
        'gold_docs': gold_docs,
        'gold_answer': example['answer']['value']
    }


def load_trivia_qa(document_save_path=None, train_data_save_path=None):
    # if save_path is not None, we save the post-processed data to the path
    full_dataset = load_dataset("trivia_qa", 'rc')
    
    processed_splits = {
        "train": [],
        "validation": [],
        "test": []
    }
    all_documents = []
    seen_doc_id = set()

    for split in processed_splits.keys():
        full_split = full_dataset[split]
        for example in tqdm(full_split, desc=f"Processing {split} split"):
            processed_example = _process_trivia_qa(example)
            processed_splits[split].append(processed_example)
            ## process the document db
            for doc in processed_example['gold_docs']:
                if doc.metadata['doc_id'] not in seen_doc_id:
                    all_documents.append(doc)
                    seen_doc_id.add(doc.metadata['doc_id'])
            for doc in processed_example['hard_neg_docs']:
                if doc.metadata['doc_id'] not in seen_doc_id:
                    all_documents.append(doc)
                    seen_doc_id.add(doc.metadata['doc_id'])
    
    if train_data_save_path is not None:
        # save
        for split, examples in processed_splits.items():
            save_path = os.path.join(train_data_save_path, f"{split}_w_qa.jsonl")
            with jsonlines.open(save_path, "w") as writer:
                writer.write_all(examples)
        print(f"Saved processed data to {train_data_save_path} folder")
    if document_save_path is not None:
        save_path = os.path.join(document_save_path, "all_documents.pkl")
        with open(save_path, "wb") as writer:
            pickle.dump(all_documents, writer)
        print(f"Saved processed documents to {save_path}")
    
    return processed_splits, all_documents


def _process_natural_questions(example: dict):
    question = example['question']['text']
    ### process documents
    ## all search context are good results
    gold_docs = []

    
    title = example['document']['title']
    url = example['document']['url']
    texts = " ".join(example['document']['tokens']['token'])
    html = example['document']['html']

    file_content = f"""
    title: {title}
    texts: {texts}
    """.replace("    ","").strip()

    doc = Document(
        page_content=file_content,
        metadata={
            "url": url,
            "source": url,
            "doc_id": url,
            "html": html
        }
    )
    gold_docs.append(doc)
    
    ## answer
    if len(example['annotations']['short_answers'][0]['text']) == 0:
        answer = "sorry, answer is not found in the document"
    else:
        answer = example['annotations']['short_answers'][0]['text'][0]
    return {
        'chat_history': [],
        'hard_neg_docs': [],
        'question': question,
        'gold_docs': gold_docs,
        'gold_answer': answer
    }


def load_natural_questions(document_save_path=None, train_data_save_path=None):
    # if save_path is not None, we save the post-processed data to the path
    full_dataset = load_dataset("natural_questions")
    
    processed_splits = {
        "train": [],
        "validation": [],
        "test": []
    }
    all_documents = []
    seen_doc_id = set()

    for split in processed_splits.keys():
        
        #  nq does not have a test set, we split by half
        if split == "validation":
            val_size = len(full_dataset['validation'])
            indices = range(0, val_size//2)
            full_split = full_dataset['validation'].select(indices)
        elif split == "test":
            print("[INFO] both validation and test set is split from the validation set because there is no test split")
            val_size = len(full_dataset['validation'])
            indices = range(val_size//2, val_size)
            full_split = full_dataset['validation'].select(indices)
        else:
            # train exists already
            full_split = full_dataset[split]
        
        for example in tqdm(full_split, desc=f"Processing {split} split"):
            processed_example = _process_natural_questions(example)
            processed_splits[split].append(processed_example)
            ## process the document db
            for doc in processed_example['gold_docs']:
                if doc.metadata['doc_id'] not in seen_doc_id:
                    all_documents.append(doc)
                    seen_doc_id.add(doc.metadata['doc_id'])
            for doc in processed_example['hard_neg_docs']:
                if doc.metadata['doc_id'] not in seen_doc_id:
                    all_documents.append(doc)
                    seen_doc_id.add(doc.metadata['doc_id'])
    
    if train_data_save_path is not None:
        # save
        for split, examples in processed_splits.items():
            save_path = os.path.join(train_data_save_path, f"{split}_w_qa.jsonl")
            with jsonlines.open(save_path, "w") as writer:
                writer.write_all(examples)
        print(f"Saved processed data to {train_data_save_path} folder")
    if document_save_path is not None:
        save_path = os.path.join(document_save_path, "all_documents.pkl")
        with open(save_path, "wb") as writer:
            pickle.dump(all_documents, writer)
        print(f"Saved processed documents to {save_path}")
    
    return processed_splits, all_documents


def _process_ms_marco(example: dict):
    question = example['query']
    ### process documents
    ## gold_doc is the selected one
    gold_docs = []
    hard_neg_docs = []
    num_docs = len(example['passages'])

    for i in range(num_docs):
        is_selected = example['passages']['is_selected'][i]
        
        text = example['passages']['passage_text'][i]
        url = example['passages']['url'][i]
        
        doc = Document(
            page_content=text,
            metadata={
                "url": url,
                "source": url,
                "doc_id": url + "\n" + text,
            }
        )
        if is_selected == 1:
            gold_docs.append(doc)
        else:
            hard_neg_docs.append(doc)
    
    ## answer
    answer = example['answers'][0]
    return {
        'chat_history': [],
        'hard_neg_docs': hard_neg_docs,
        'question': question,
        'gold_docs': gold_docs,
        'gold_answer': answer
    }


def load_ms_marco(document_save_path=None, train_data_save_path=None):
    # if save_path is not None, we save the post-processed data to the path
    full_dataset = load_dataset("ms_marco", "v2.1")
    
    processed_splits = {
        "train": [],
        "validation": [],
        "test": []
    }
    all_documents = []
    seen_doc_id = set()

    for split in processed_splits.keys():
        #  ms_marco test set has NO ANSWERs. So we just split the validation set
        if split == "validation":
            val_size = len(full_dataset['validation'])
            indices = range(0, val_size//2)
            full_split = full_dataset['validation'].select(indices)
        elif split == "test":
            print("[INFO] both validation and test set is split from the validation set because the test set has no answers")
            val_size = len(full_dataset['validation'])
            indices = range(val_size//2, val_size)
            full_split = full_dataset['validation'].select(indices)
        else:
            # train exists already
            full_split = full_dataset[split]
        
        for example in tqdm(full_split, desc=f"Processing {split} split"):
            processed_example = _process_ms_marco(example)
            processed_splits[split].append(processed_example)
            ## process the document db
            for doc in processed_example['gold_docs']:
                if doc.metadata['doc_id'] not in seen_doc_id:
                    all_documents.append(doc)
                    seen_doc_id.add(doc.metadata['doc_id'])
            for doc in processed_example['hard_neg_docs']:
                if doc.metadata['doc_id'] not in seen_doc_id:
                    all_documents.append(doc)
                    seen_doc_id.add(doc.metadata['doc_id'])
    
    if train_data_save_path is not None:
        # save
        for split, examples in processed_splits.items():
            save_path = os.path.join(train_data_save_path, f"{split}_w_qa.jsonl")
            with jsonlines.open(save_path, "w") as writer:
                writer.write_all(examples)
        print(f"Saved processed data to {train_data_save_path} folder")
    if document_save_path is not None:
        save_path = os.path.join(document_save_path, "all_documents.pkl")
        with open(save_path, "wb") as writer:
            pickle.dump(all_documents, writer)
        print(f"Saved processed documents to {save_path}")
    
    return processed_splits, all_documents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default="trivia_qa",
        choices=["trivia_qa", "natural_questions", "ms_marco"],
        help="The dataset to load"
    )
    parser.add_argument(
        "--document_save_path", type=str,
        default=None,
        help="The path to save the processed documents"
    )
    parser.add_argument(
        "--train_data_save_path", type=str,
        default=None,
        help="The path to save the processed data"
    )
    args = parser.parse_args()

    if args.dataset == "trivia_qa":
        load_trivia_qa(
            document_save_path=args.document_save_path,
            train_data_save_path=args.train_data_save_path
        )
    elif args.dataset == "natural_questions":
        load_natural_questions(
            document_save_path=args.document_save_path,
            train_data_save_path=args.train_data_save_path
        )
    elif args.dataset == "ms_marco":
        load_ms_marco(
            document_save_path=args.document_save_path,
            train_data_save_path=args.train_data_save_path
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")