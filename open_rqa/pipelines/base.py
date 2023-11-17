from langchain.schema import Document, BaseRetriever
from langchain.memory import ConversationBufferMemory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from retry import retry
from open_rqa.schema.outputs import GenerationOutput
from open_rqa.schema.outputs import RetrievalOutput, GenerationOutput
from open_rqa.embeddings.base import EmbeddingsWrapper
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
	def generate_from_docs(self, batch_inputs, retr_output: RetrievalOutput) -> GenerationOutput:
		raise NotImplementedError
	
	@abstractmethod
	def answer_guardrail(self, raw_gen_output: GenerationOutput) -> GenerationOutput:
		raise NotImplementedError


class TamarinPipeline(RetrievalQAModel):
	dont_know_reply = "I'm sorry, I don't know the answer to your question. I will ping your manager Arbit Chen to answer your question and get back to you later!"

	def __init__(self, texts_db_path):
		self._texts_db_path = texts_db_path
		self.embeddings: EmbeddingsWrapper = None  # type: ignore
		return
	
	@property
	def texts_db_path(self):
		return self._texts_db_path
	
	@property
	def memory(self) -> ConversationBufferMemory:
		raise NotImplementedError
	
	@memory.setter
	def memory(self, value: ConversationBufferMemory):
		raise NotImplementedError
	
	@property
	def retriever(self) -> BaseRetriever:
		raise NotImplementedError
	
	def _load_text_from_db(self) -> List[Document]:
		with open(self.texts_db_path, 'rb') as f:
			documents = pickle.load(f)
		print(f"Loaded {len(documents)} documents")
		return documents
	
	def _load_memory(self, chat_history: List[Dict[str, str]]):
		memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
		for chat in chat_history:
			memory.chat_memory.add_user_message(chat['question'])
			memory.chat_memory.add_ai_message(chat['answer'])
		return memory
	
	def _init_dummy_memory(self):
		memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
		memory.chat_memory.add_user_message("Hello")
		memory.chat_memory.add_ai_message("Hi")
		return memory
	
	@abstractmethod
	def __call__(self, inputs: dict) -> Dict[str, Any]:
		"""given an input question, do retrieval and answer generation
		"""
		raise NotImplementedError
	
	def _tweak_ai_memory(self, answer_str):
		self.memory.buffer[-1].content = answer_str
		return
	
	def _format_doc_data(self, doc):
		return {
			"source": doc.metadata["title"],
			"source_url": doc.metadata["source"],
			"source_type": doc.metadata["mimetype"],
			# "content": doc.page_content,
		}
	
	def _process_dontknow(self, ori_answer, ori_documents):
		really_dontknow = False

		answer_lower = ori_answer.lower()
		if "I'm sorry".lower() in answer_lower or "don't know".lower() in answer_lower:
			really_dontknow = True
		if "[DONTKNOW]" in ori_answer:
			ori_answer = ori_answer.replace("[DONTKNOW]", "").strip()
			# really don't know
			if len(ori_answer) < 10 or "I'm sorry".lower() in answer_lower or "don't know".lower() in answer_lower:
				really_dontknow = True
		
		# still some wierd cases
		cleaned_paragraphs = []
		for paragraph in ori_answer.split('\n'):
			cleaned_sents = []
			for sent in nltk.sent_tokenize(paragraph):
				if 'DONTKNOW' in sent:
					continue
				cleaned_sents.append(sent)
			cleaned_paragraphs.append(' '.join(cleaned_sents))
		ori_answer = '\n'.join(cleaned_paragraphs)

		# if result is gone due to don't know, or there is just no source documents
		if really_dontknow or ori_answer == '' or len(ori_documents) == 0:
			ori_answer = self.dont_know_reply
			really_dontknow = True

		cleaned_answer = ori_answer.strip()
		cleaned_answer = re.sub(r'\[ .*\]', '', cleaned_answer).strip()
		self._tweak_ai_memory(cleaned_answer)

		# if really don't know, 
		# 1. tweak the memory so that it is more stable in the future
		# 2. remove the source documents
		cleaned_documents = ori_documents
		if really_dontknow:
			new_answer_str = cleaned_answer + ' [DONTKNOW]'
			cleaned_documents = []
			self._tweak_ai_memory(new_answer_str)
		return cleaned_answer, cleaned_documents
	
	def _check_sources(self, answer: str, source_documents: List[Document]):
		return source_documents
		# TODO: simple TFIDF just gives high scores hence is hard to filter. We probably need semantic checks.
		corpus = [f"document title: {doc.metadata['title']}.\ncontent: {doc.page_content}" for doc in source_documents]
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(corpus)
		similarlity = cosine_similarity(vectorizer.transform([answer]), X)
		return source_documents
	
	def answer_guardrail(self, raw_gen_output: GenerationOutput) -> GenerationOutput:
		# remove sources for dontknow answers
		new_answers = []
		new_docs = []
		for i, answer in enumerate(raw_gen_output.generated_answers):
			logger.info('raw answer: ' + answer)

			source_doc = raw_gen_output.source_documents[i]
			if answer.startswith('Assistant:'):
				answer = answer[len('Assistant:'):].strip()
			cleaned_ans, cleaned_sources = self._process_dontknow(answer, source_doc)
			cleaned_sources = self._check_sources(cleaned_ans, cleaned_sources)
			
			logger.info('final answer: ' + cleaned_ans)

			new_answers.append(cleaned_ans)
			new_docs.append(cleaned_sources)
		
		return GenerationOutput(
			generated_answers=new_answers,
			source_documents=new_docs,
		)
	
	def _prepare_output(self, result: GenerationOutput):
		# reformat the documents for frontend
		formatted_src_docs = []
		for docs in result.source_documents:
			fmt_docs = [self._format_doc_data(doc) for doc in docs]
			formatted_src_docs.append(fmt_docs)
		
		output: Dict[str, List] = {
			"answer": result.generated_answers,
			"source_documents": formatted_src_docs
		}
		# if single batch, remove the batch dimension
		if len(output['answer']) == 1:
			output = {k: v[0] for k, v in output.items()}

		# TODO: this only makes sense in a non-batched setting
		hidden_answer = self.memory.buffer[-1].content
		output['hidden_answer'] = hidden_answer  # the 'real' answer, used by external memory management
		return output
	

class APIConvQAModel(TamarinPipeline):
	def __init__(self, api: str):
		self.base_api = api
		return
	
	def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
		raise ValueError("APIConvQAModel does not support build_index")
	
	def retrieve(self, batch_inputs, passages=None, indices=None) -> RetrievalOutput:
		raise ValueError("APIConvQAModel does not support retrieval")
	
	@retry(Exception, tries=3, delay=2)
	def gen_from_docs_wrapper(self, inputs: dict) -> Dict[str, Any]:
		api = self.base_api + '/gen/beta'

		logger.info(f"Calling {api} with {inputs}")
		response = requests.post(api, json=inputs)
		return response.json()
	
	def generate_from_docs(self, batch_inputs, retr_output: RetrievalOutput) -> GenerationOutput:
		raise ValueError("APIConvQAModel does not support generate_from_docs, use __call__ instead")

	@retry(Exception, tries=3, delay=2)
	def __call__(self, inputs: dict) -> Dict[str, Any]:
		api = self.base_api + '/qa/beta'

		logger.info(f"Calling {api} with {inputs}")
		response = requests.post(api, json={'inputs': inputs})
		return response.json()