from langchain.schema import Document, BaseRetriever
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from retry import retry

from open_rqa.retrievers.base import RetrievalOutput
from open_rqa.schema.dialogue import RQAOutput
from open_rqa.train.embeddings import EmbeddingsWrapper, LocalEmbeddings, LocalBERTMLMEmbeddings
import torch
import nltk
import pickle
import requests
import re
import logging

logger = logging.getLogger(__name__)


class RetrievalModel(ABC):
	embeddings: Optional[EmbeddingsWrapper] = None

	def __init__(self, *args, **kwargs):
		return
	
	@abstractmethod
	def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
		raise NotImplementedError
	
	@abstractmethod
	def retrieve(self, batch_inputs, documents=None, indices=None) -> RetrievalOutput:
		raise NotImplementedError

class RetrievalQAModel(RetrievalModel):
	@abstractmethod
	def generate_from_docs(self, batch_inputs, retr_output: RetrievalOutput) -> RQAOutput:
		raise NotImplementedError
	
	@abstractmethod
	def answer_guardrail(self, raw_gen_output: RQAOutput) -> RQAOutput:
		raise NotImplementedError

class RetrieverfromBertModel(RetrievalModel):
	def __init__(self, model, tokenizer, search_kwargs):
		self.model = model
		self.tokenizer = tokenizer
		self.search_args = search_kwargs
		self.embeddings: LocalEmbeddings = None
		return
	
	def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
		embeddings = LocalEmbeddings(self.model, self.tokenizer, index_path=None, device = "cuda")

		# asuming we are doing an inner product search, then normalize_L2=False
		normalize_L2 = self.search_args['search_kwargs'].pop('normalize_L2', False)
		docsearch = FAISS.from_documents(documents, embeddings)  # this will call from_texts under the hood
		self.embeddings = embeddings
		
		# self.retriever = docsearch.as_retriever(**self.search_args)
		self.retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={'k': 4})
		embedding_tensor = torch.tensor(self.embeddings.document_embeddings)
		return embedding_tensor

		# # now build index manually
		# passages = [format_str.format(title=doc.metadata['title'], text=doc.page_content) for doc in documents]

		# if len(passages) == 0:
		# 	return torch.tensor([])
		# list_embeddings = self.embeddings.build_index_from_texts(passages)
		# # [len(passages), 768]
		# doc_embeddings = torch.tensor(list_embeddings)
		# return doc_embeddings
	
	def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
		docs = self.retriever.get_relevant_documents(question)
		return docs
	
	def _batch_get_docs(self, questions: List[str], inputs: Dict[str, Any]) -> List[List[Document]]:
		all_docs = []
		for question in questions:
			docs = self._get_docs(question, inputs)
			all_docs.append(docs)
		return all_docs
	
	def retrieve(self, batch_inputs, passages=None, indices=None) -> RetrievalOutput:
		questions = batch_inputs["question"]
		chat_history_strs = batch_inputs["chat_history_str"]

		# retrieve
		docs = self._batch_get_docs(questions, {})
		output = RetrievalOutput(
            batch_source_documents=docs
        )
		return output
	

class RetrieverfromBertMLMModel(RetrieverfromBertModel):
	def __init__(self, model, tokenizer, search_kwargs):
		self.model = model
		self.tokenizer = tokenizer
		self.search_args = search_kwargs
		self.embeddings: LocalBERTMLMEmbeddings = None
		return
	
	def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
		embeddings = LocalBERTMLMEmbeddings(self.model, self.tokenizer, index_path=None, device = "cuda")

		# asuming we are doing an inner product search, then normalize_L2=False
		normalize_L2 = self.search_args['search_kwargs'].pop('normalize_L2', False)
		docsearch = FAISS.from_documents(documents, embeddings)
		self.embeddings = embeddings
		
		self.retriever = docsearch.as_retriever(**self.search_args)
		embedding_tensor = torch.tensor(self.embeddings.document_embeddings)
		return embedding_tensor