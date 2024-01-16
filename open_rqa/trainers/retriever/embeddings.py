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


def batch_iterator(dset, batch_size, drop_last=False, shuffle=False):
	batch = []
	for item in dset:
		batch.append(item)
		if len(batch) == batch_size:
			yield batch
			batch = []
	if len(batch) > 0 and not drop_last:
		yield batch

def mean_pooling(token_embeddings, mask):
	token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
	sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
	return sentence_embeddings

def compute_embedding(encoded_inputs, outputs):
	embedding = mean_pooling(outputs.last_hidden_state, encoded_inputs['attention_mask'])
	return embedding

def embed_document_batch(tokenizer, model, batch, batch_size=8, to_list=False):
	b = batch_iterator(batch, batch_size=batch_size, shuffle=False)
	embeddings = []
	for bb in b:
		encoded_inputs = tokenizer(
			bb, return_tensors="pt",
			padding='max_length', max_length=512,
			truncation=True
		)
		for k, v in encoded_inputs.items():
			encoded_inputs[k] = v.to(model.device)
		outputs = model(**encoded_inputs, output_hidden_states=True)
		# [len(texts), 768)]
		embedding = compute_embedding(encoded_inputs, outputs)
		if to_list:
			embeddings.extend(embedding.tolist())
		else:
			embeddings.append(embedding)
	if not to_list:
		embeddings = torch.concat(embeddings, dim=0)
	return embeddings


class LocalEmbeddings(Embeddings):
	def __init__(self, model, tokenizer, device = "cuda:0"):
		model.eval()
		model.to(device)
		self.device = device
		self.model = model
		self.tokenizer = tokenizer
		return

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		embeddings = embed_document_batch(self.tokenizer, self.model, texts, to_list=True)
		return embeddings

	def embed_query(self, text) -> List[float]:
		return self.embed_documents([text])[0]
