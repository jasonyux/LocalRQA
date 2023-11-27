from typing import List
from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import (
    InMemoryStore,
    LocalFileStore,
    RedisStore,
    UpstashRedisStore,
)
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


from open_rqa.schema.dialogue import DialogueSession
from open_rqa.schema.document import Document
from open_rqa.retrievers.base import BaseRetriever
from open_rqa.retrievers.base import RetrievalOutput
# from open_rqa.vectorstore.faiss import MyFAISS


class FaissRetriever(BaseRetriever):

	def __init__(
		self, 
		texts: List[Document], 
		embeddings=OpenAIEmbeddings(model='text-embedding-ada-002'), 
		index_path="./index"
	) -> None:
		"""

		Args:
			texts (List[Document]): documents for retriever
			embeddings (_type_, optional): embeddings wrapper supported by LangChain. Defaults to OpenAIEmbeddings(model='text-embedding-ada-002').
			index_path (str, optional): saving index path. Defaults to "./index".
		"""
		
		fs = LocalFileStore(index_path)
		cached_embedder = CacheBackedEmbeddings.from_bytes_store(
			embeddings, fs, namespace=embeddings.model
		)
		super().__init__(texts, cached_embedder)
		self.retriever = self._init_retriever()
	
	def _init_retriever(self, **kwargs):
		"""initiate FAISS retriever

		Returns:
			_type_: retriever
		"""

		docsearch = FAISS.from_documents(self.texts, self.embeddings)
		
		retriever = docsearch.as_retriever(**kwargs)
		return retriever
	
	def retrieve(
		self,
		question_generator=None,
		*args, **kwargs
	) -> RetrievalOutput:
		"""given a batched query and dialogue history, retrieve relevant documents

		Args:
			batch_query (List[str]): _description_
			batch_dialogue_history (List[DialogueSession]): _description_

		Returns:
			RetrievalOutput: _description_
		"""
		batch_query = kwargs.get('batch_query')
		batch_dialogue_history = kwargs.get('batch_dialogue_history')
		all_docs = []

		for i in range(len(batch_query)):
			question = batch_query[i]
			chat_history_str = batch_dialogue_history[i].to_string()
			docs = self.retriever.get_relevant_documents(question)

			all_docs.append(docs)


		output = RetrievalOutput(
			batch_source_documents=all_docs
		)
		return output
		
