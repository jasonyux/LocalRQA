from typing import List, Dict, Union, Callable
from transformers import AutoTokenizer, BatchEncoding
from tqdm.auto import tqdm
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.qa_llms.prompts import RQA_PROMPT, RQA_PROMPT_TRAIN
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
        sep_user = " ",
        sep_sys = "</s>",
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
        self.sep_user = sep_user
        self.sep_sys = sep_sys

        flattened_formatted_data = self.prepare_data(qa_w_doc_data)
        self.data = self.encode_data(flattened_formatted_data)
        if shuffle:
            # usually the training data files are ALREADY shuffled
            # in the case of few shot experiments, we want to explicitly shuffle the data
            random.seed(42)
            random.shuffle(self.data)
        return
    
    def prepare_data(self, qa_w_doc_data: List[Dict]):
        _necessary_fields = ['question', 'chat_history', 'gold_answer', 'gold_docs']
        assert all([field in qa_w_doc_data[0].keys() for field in _necessary_fields]), \
            f"Missing necessary fields in qa_w_doc_data: {qa_w_doc_data[0].keys()}"
        
        formatted_data = []
        for i, sample in enumerate(qa_w_doc_data):
            gold_docs = [Document.from_dict(doc) for doc in sample['gold_docs']]
            chat_history = sample['chat_history']
            question = sample['question']
            gold_answer = sample['gold_answer']
            # format dialogue
            dialogue_session = DialogueSession.from_list(chat_history)
            dialogue_session.sep_sys = self.sep_sys
            dialogue_session.sep_user = self.sep_user
            dialogue_session.assistant_prefix = self.assistant_prefix
            dialogue_session.user_prefix = self.user_prefix
            dialogue_session.add_user_message(question)
            dialogue_session.add_system_message(
                system_message=gold_answer,
                source_documents=gold_docs
            )
            fmt_dialogue = dialogue_session.to_string()

            # prompt with retrieved documents
            formatted_gold_docs_string = "\n".join([doc.fmt_content for doc in gold_docs])
            fmt_prompt = RQA_PROMPT_TRAIN.format(
                formatted_documents=formatted_gold_docs_string,
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
        sep_user = " ",
        sep_sys = "</s>",
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
        self.sep_user = sep_user
        self.sep_sys = sep_sys

        flattened_formatted_data = self.prepare_data(qa_w_doc_data)
        self.data = self.encode_data(flattened_formatted_data)
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        return

    def _format_retrieved_docs(self, gold_docs: List[Document], retrieved_docs: List[Document]):
        max_num_docs = self.max_num_to_retrieve + 1
        formatted_docs = [doc.fmt_content.strip() for doc in gold_docs][:max_num_docs]
        for doc in retrieved_docs:
            if len(formatted_docs) == max_num_docs:
                break
            if doc.fmt_content in set(formatted_docs):
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
        _necessary_fields = ['question', 'chat_history', 'gold_answer', 'gold_docs']
        assert all([field in qa_w_doc_data[0].keys() for field in _necessary_fields]), \
            f"Missing necessary fields in qa_w_doc_data: {qa_w_doc_data[0].keys()}"
        
        ## init retriever
        all_docs = []
        for sample in qa_w_doc_data:
            all_docs.extend([Document.from_dict(doc) for doc in sample['gold_docs']])
        retriever: BaseRetriever = self.retriever_init_fn(
            embedding_model=self.embeddings,
            documents=all_docs,
        )
        all_retrieved_docs = self.pre_retrieve_all_docs(
            retriever=retriever,
            all_questions=[sample['question'] for sample in qa_w_doc_data]
        )
        
        formatted_data = []
        for i, sample in enumerate(qa_w_doc_data):
            gold_docs = [Document.from_dict(doc) for doc in sample['gold_docs']]
            chat_history = sample['chat_history']
            question = sample['question']
            gold_answer = sample['gold_answer']
            retrieved_docs = all_retrieved_docs[i]
            # format dialogue
            dialogue_session = DialogueSession.from_list(chat_history)
            dialogue_session.sep_sys = self.sep_sys
            dialogue_session.sep_user = self.sep_user
            dialogue_session.assistant_prefix = self.assistant_prefix
            dialogue_session.user_prefix = self.user_prefix
            dialogue_session.add_user_message(question)
            dialogue_session.add_system_message(
                system_message=gold_answer,
                source_documents=gold_docs
            )
            fmt_dialogue = dialogue_session.to_string()

            # prompt with retrieved documents
            fmt_retrieved_docs_w_gold = self._format_retrieved_docs(gold_docs, retrieved_docs)
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
        logger.info(f"Processed {len(encoded_data)} RQA inputs")
        return encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SupervisedFiDRQAwRetrieverDataset(torch.utils.data.Dataset):
    """train FiD with (a fixed) retriever

    Args:
        torch (_type_): _description_
    """
    def __init__(
        self,
        qa_w_doc_data: List[Dict],
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
        retriever_init_fn: Callable,
        tokenizer: AutoTokenizer,
        assistant_prefix: str = "ASSISTANT",
        user_prefix: str = "USER",
        max_num_to_retrieve: int = 3,
        encoder_max_length=512,  # since its FiD, we don't concatenate the documents
        decoder_max_length=256,
        start_data_idx=0,
        end_data_idx=None,
        shuffle=False
    ):
        self.embeddings = embedding_model
        self.retriever_init_fn = retriever_init_fn
        self.max_num_to_retrieve = max_num_to_retrieve  # excluding gold doc
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.start_data_idx = start_data_idx
        self.end_data_idx = end_data_idx
        self.assistant_prefix = assistant_prefix
        self.user_prefix = user_prefix

        flattened_input, flattened_output = self.prepare_data(qa_w_doc_data)
        self.data = self.encode_data(flattened_input, flattened_output)
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        return

    def pre_retrieve_all_docs(self, retriever: BaseRetriever, all_questions: List[str]):
        # the queries are searched one by one anyway
        all_retrieved_docs = retriever.retrieve(
            batch_questions=all_questions
        ).batch_source_documents
        return all_retrieved_docs

    def _combine_retrieved_docs(self, gold_docs: List[Document], retrieved_docs: List[Document]):
        max_num_docs = self.max_num_to_retrieve + 1
        combined_docs = []
        seen_doc_content = set()
        for doc in gold_docs:
            if doc.fmt_content not in seen_doc_content:
                combined_docs.append(doc)
                seen_doc_content.add(doc.fmt_content)
        for doc in retrieved_docs:
            if doc.fmt_content not in seen_doc_content:
                combined_docs.append(doc)
                seen_doc_content.add(doc.fmt_content)
        
        # FiD requires all batches have the same number of documents
        if len(combined_docs) > max_num_docs:
            combined_docs = combined_docs[:max_num_docs]
        else:
            # we pad
            num_to_pad = max_num_docs - len(combined_docs)
            combined_docs += [combined_docs[-1]] * num_to_pad
        return combined_docs

    def encode_fid_inputs(self, q_w_passages: List[str], max_length):
        p = self.tokenizer.batch_encode_plus(
            q_w_passages,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        # 3D input, so that during training we have (batch_size, num_passages, max_length)
        passage_ids = p['input_ids']
        passage_masks = p['attention_mask']
        return BatchEncoding({
            'input_ids': passage_ids,
            'attention_mask': passage_masks.bool()
        })

    def prepare_data(self, qa_w_doc_data: List[Dict]):
        _necessary_fields = ['question', 'chat_history', 'gold_answer', 'gold_docs']
        assert all([field in qa_w_doc_data[0].keys() for field in _necessary_fields]), \
            f"Missing necessary fields in qa_w_doc_data: {qa_w_doc_data[0].keys()}"
        
        ## init retriever
        all_docs = []
        for sample in qa_w_doc_data:
            all_docs.extend([Document.from_dict(doc) for doc in sample['gold_docs']])
        retriever: BaseRetriever = self.retriever_init_fn(
            embedding_model=self.embeddings,
            documents=all_docs,
        )
        all_retrieved_docs = self.pre_retrieve_all_docs(
            retriever=retriever,
            all_questions=[sample['question'] for sample in qa_w_doc_data]
        )
        
        formatted_input_data = []
        formatted_output_data = []
        for i, sample in enumerate(qa_w_doc_data):
            gold_docs = [Document.from_dict(doc) for doc in sample['gold_docs']]
            chat_history = sample['chat_history']
            question = sample['question']
            gold_answer = sample['gold_answer'] + " </s>"
            retrieved_docs = all_retrieved_docs[i]
            # format dialogue
            dialogue_session = DialogueSession.from_list(chat_history)
            dialogue_session.assistant_prefix = self.assistant_prefix
            dialogue_session.user_prefix = self.user_prefix
            dialogue_session.add_user_message(question)
            # since FiD is encoder decoder, input do NOT include the answer
            fmt_dialogue = dialogue_session.to_string()

            ### prompt with retrieved documents
            # fid does it in parallel
            to_include_docs = self._combine_retrieved_docs(gold_docs, retrieved_docs)
            fid_input = []
            for doc in to_include_docs:
                # since FiD is encoder decoder, input do NOT include the answer
                prompt = RQA_PROMPT.format(
                    formatted_documents = doc.fmt_content,
                    formatted_chat = fmt_dialogue,
                    assistant_prefix = self.assistant_prefix,
                )
                fid_input.append(prompt)
            formatted_input_data.append(fid_input)
            # fid output
            formatted_output_data.append(gold_answer)
        # print one example data
        logger.info("Example formatted data:")
        logger.info(formatted_input_data[0])
        logger.info(formatted_output_data[0])
        return formatted_input_data, formatted_output_data
    
    def encode_data(self, encoder_text_data: List[List[str]], decoder_text_data: List[str]):
        encoded_data = []

        assert len(encoder_text_data) == len(decoder_text_data)
        for idx in tqdm(range(len(encoder_text_data))[self.start_data_idx:self.end_data_idx], desc="Encoding data"):
            encoder_input = encoder_text_data[idx]
            decoder_input = decoder_text_data[idx]

            tokenized = self.encode_fid_inputs(encoder_input, max_length=self.encoder_max_length)
            # tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

            decoder_tokenized = self.tokenizer(
                decoder_input,
                max_length=self.decoder_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized["labels"] = decoder_tokenized["input_ids"].squeeze(0).clone()

            encoded_data.append(tokenized)
        logger.info(f"Processed {len(encoded_data)} FiD inputs")
        return encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]