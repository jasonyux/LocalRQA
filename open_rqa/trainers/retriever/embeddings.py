from abc import abstractmethod
from typing import List
import math
import os
import logging
import fnmatch

from tqdm.auto import tqdm
import torch
from langchain.embeddings.base import Embeddings

from open_rqa.trainers import dist_utils


logger = logging.getLogger(__name__)


class EmbeddingsWrapper(Embeddings):
	document_embeddings = None

	@abstractmethod
	def build_index_from_texts(self, texts: List[str]):
		raise NotImplementedError


class LocalEmbeddings(EmbeddingsWrapper):
	def __init__(self, model, tokenizer, index_path = None, device = "cuda:0"):
		model.eval()
		model.to(device)
		self.device = device
		self.model = model
		self.tokenizer = tokenizer
		self.index_path = index_path

		self.document_embeddings = None
		return

	def mean_pooling(self, token_embeddings, mask):
		token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
		sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
		return sentence_embeddings
	
	def batch_iterator(self, data_iterator, batch_size=16):
		batch = []
		batch_counter = 0
		for example in data_iterator:
			batch.append(example)
			if len(batch) == batch_size:
				batch_counter += 1
				yield batch
				batch = []
		if len(batch) > 0:
			yield batch
	
	def __embed_document_batch(self, batch):
		for k, v in batch.items():
			batch[k] = v.to(self.device)
		outputs = self.model(**batch)
		# [len(texts), 768)]
		embeddings = self.mean_pooling(outputs.last_hidden_state, batch['attention_mask'])
		return embeddings.tolist()
	
	def load_index(self):
		"""
		Loads sharded embeddings
		"""
		total_saved_shards = 0
		for _, _, filenames in os.walk(self.index_path):
			for _ in fnmatch.filter(filenames, 'embeddings.*.pt'):
				total_saved_shards += 1
		logger.info(f"Found total saved shards: {total_saved_shards}")
		rank = dist_utils.get_rank()
		ws = dist_utils.get_world_size()
		assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
		shards_per_worker = total_saved_shards // ws
		embeddings = []
		for shard_id in tqdm(range(rank * shards_per_worker, (rank + 1) * shards_per_worker), desc="Loading index shards"):
			embeddings_shard_path = os.path.join(self.index_path, f"embeddings.{shard_id}.pt")
			embeddings.append(torch.load(embeddings_shard_path, map_location="cpu"))
		
		embeddings_t = torch.concat(embeddings, dim=1)
		embeddings_t = embeddings_t.T.tolist()
		return embeddings_t
	
	def build_index_from_texts(self, texts: List[str]):
		batch_size = 8
		num_batches = math.ceil(len(texts) / batch_size)
		batches = self.batch_iterator(texts, batch_size)
		embeddings = []
		for batch in tqdm(batches, total=num_batches, desc="Embedding documents"):
			inputs = self.tokenizer(batch, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
			batch_embedding = self.__embed_document_batch(inputs)
			embeddings.extend(batch_embedding)
		return embeddings

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		if self.document_embeddings is not None:
			return self.document_embeddings

		if self.index_path is not None:
			print(f"Loading index from {self.index_path}")
			embeddings = self.load_index()
		else:
			embeddings = self.build_index_from_texts(texts)
		self.document_embeddings = embeddings
		return embeddings

	def embed_query(self, text) -> List[float]:
		inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
		batch_embedding = self.__embed_document_batch(inputs)
		return batch_embedding[0]
