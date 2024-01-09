from typing import List, Dict, Union, Callable
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.qa_llms.prompts import RQA_PROMPT_TRAIN
from open_rqa.retrievers.base import BaseRetriever
from open_rqa.retrievers.faiss_retriever import FaissRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
import random
import logging


logger = logging.getLogger(__name__)


def batch_iterator(dset, batch_size, drop_last=False):
    batch = []
    for item in dset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch


class NoopDataCollator:
    def __call__(self, features):
        return features


class SupervisedRQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        qa_w_doc_data: List[Dict],
        tokenizer: AutoTokenizer,
        assistant_prefix: str = "ASSISTANT",
        user_prefix: str = "USER",
        max_length=650,  # should be enough for one 400 token passgae + answer
        start_data_idx=0,
        end_data_idx=None,
        shuffle=False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_data_idx = start_data_idx
        self.end_data_idx = end_data_idx
        self.assistant_prefix = assistant_prefix
        self.user_prefix = user_prefix

        flattened_formatted_data = self.prepare_data(qa_w_doc_data)
        self.data = self.encode_data(flattened_formatted_data)
        if shuffle:
            # usually the training data files are ALREADY shuffled
            # in the case of few shot experiments, we want to explicitly shuffle the data
            random.seed(42)
            random.shuffle(self.data)
        return
    
    def prepare_data(self, qa_w_doc_data: List[Dict]):
        _necessary_fields = ['question', 'chat_history_str', 'gold_answer', 'gold_doc']
        assert(all([field in qa_w_doc_data[0].keys() for field in _necessary_fields]))
        
        formatted_data = []
        for sample in qa_w_doc_data:
            gold_doc: Document = sample['gold_doc']
            chat_history_str = sample['chat_history_str']
            question = sample['question']
            gold_answer = sample['gold_answer']
            # format dialogue
            dialogue_session = DialogueSession.from_list(chat_history_str)
            dialogue_session.assistant_prefix = self.assistant_prefix
            dialogue_session.user_prefix = self.user_prefix
            dialogue_session.add_user_message(question)
            dialogue_session.add_system_message(
                system_message=gold_answer,
                source_documents=[gold_doc]
            )
            fmt_dialogue = dialogue_session.to_string()

            # gold prompt
            fmt_prompt = RQA_PROMPT_TRAIN.format(
                formatted_documents=gold_doc.fmt_content,
                formatted_chat=fmt_dialogue,
            )
            formatted_data.append(fmt_prompt)
        # print two exapmle data
        logger.info("Example formatted data:")
        logger.info(formatted_data[0])
        logger.info(formatted_data[1])
        return formatted_data
    
    def encode_data(self, text_data: List[str]):
        encoded_data = []
        for text in tqdm(text_data[self.start_data_idx:self.end_data_idx], desc="Encoding data"):
            tokenized = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            tokenized["labels"] = tokenized["input_ids"].clone()

            encoded_data.append(tokenized)
        logger.info(f"Processed {len(encoded_data)} documents")
        return encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def retriever_init(embeddings_model, documents, train_index_path):
    retriever = FaissRetriever(
        documents,
        embeddings=embeddings_model,
        index_path=train_index_path
    )
    return retriever


class SupervisedRQAwRetrieverDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        qa_w_doc_data: List[Dict],
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
        retriever_init_fn: Callable,
        tokenizer: AutoTokenizer,
        assistant_prefix: str = "ASSISTANT",
        user_prefix: str = "USER",
        max_num_to_retrieve: int = 3,
        max_length=2048,  # should be enough for 4 * 400 token passgae + answer
        start_data_idx=0,
        end_data_idx=None,
        shuffle=False
    ):
        self.embeddings = embedding_model
        self.retriever_init_fn = retriever_init_fn
        self.max_num_to_retrieve = max_num_to_retrieve  # excluding gold doc
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_data_idx = start_data_idx
        self.end_data_idx = end_data_idx
        self.assistant_prefix = assistant_prefix
        self.user_prefix = user_prefix

        flattened_formatted_data = self.prepare_data(qa_w_doc_data)
        self.data = self.encode_data(flattened_formatted_data)
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        return

    def _format_retrieved_docs(self, gold_doc: Document, retrieved_docs: List[Document]):
        formatted_docs = []
        formatted_docs.append(gold_doc.fmt_content)
        for doc in retrieved_docs:
            if len(formatted_docs) == self.max_num_to_retrieve + 1:
                break
            if doc.fmt_content == gold_doc.fmt_content:
                continue
            formatted_docs.append(doc.fmt_content.strip())
        formatted_docs_string = "\n".join(formatted_docs)
        return formatted_docs_string

    def pre_retrieve_all_docs(self, retriever: BaseRetriever, all_questions: List[str]):
        # the queries are searched one by one anyway
        all_retrieved_docs = retriever.retrieve(
            batch_questions=all_questions
        ).batch_source_documents
        return all_retrieved_docs

    def prepare_data(self, qa_w_doc_data: List[Dict]):
        _necessary_fields = ['question', 'chat_history_str', 'gold_answer', 'gold_doc']
        assert(all([field in qa_w_doc_data[0].keys() for field in _necessary_fields]))
        
        retriever: BaseRetriever = self.retriever_init_fn(
            embedding_model=self.embeddings,
            documents=[Document.from_dict(sample['gold_doc']) for sample in qa_w_doc_data],
        )
        all_retrieved_docs = self.pre_retrieve_all_docs(
            retriever=retriever,
            all_questions=[sample['question'] for sample in qa_w_doc_data]
        )
        
        formatted_data = []
        for i, sample in enumerate(qa_w_doc_data):
            gold_doc = Document.from_dict(sample['gold_doc'])
            chat_history_str = sample['chat_history_str']
            question = sample['question']
            gold_answer = sample['gold_answer']
            retrieved_docs = all_retrieved_docs[i]
            # format dialogue
            dialogue_session = DialogueSession.from_list(chat_history_str)
            dialogue_session.assistant_prefix = self.assistant_prefix
            dialogue_session.user_prefix = self.user_prefix
            dialogue_session.add_user_message(question)
            dialogue_session.add_system_message(
                system_message=gold_answer,
                source_documents=[gold_doc]
            )
            fmt_dialogue = dialogue_session.to_string()

            # prompt with retrieved documents
            fmt_retrieved_docs_w_gold = self._format_retrieved_docs(gold_doc, retrieved_docs)
            fmt_prompt = RQA_PROMPT_TRAIN.format(
                formatted_documents=fmt_retrieved_docs_w_gold,
                formatted_chat_w_answer=fmt_dialogue,
            )
            formatted_data.append(fmt_prompt)
        # print one example data
        logger.info("Example formatted data:")
        logger.info(formatted_data[0])
        return formatted_data
    
    def encode_data(self, text_data: List[str]):
        encoded_data = []
        for text in tqdm(text_data[self.start_data_idx:self.end_data_idx], desc="Encoding data"):
            tokenized = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            tokenized["labels"] = tokenized["input_ids"].clone()

            encoded_data.append(tokenized)
        logger.info(f"Processed {len(encoded_data)} documents")
        return encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]