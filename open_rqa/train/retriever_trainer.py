from copy import deepcopy
from transformers import Trainer, BertModel, BertForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
from typing import Optional, List, Union, Dict, Any, Tuple, Type, Callable
from open_rqa.schema import Document
from open_rqa.train.model.wrappers import RetrievalModel
from open_rqa.evaluation.evaluator import RetrieverEvaluator, EvaluatorConfig
from open_rqa.train.utils.arguments import RetrievalQATrainingArguments
import torch
import torch.nn as nn
import random
import os
import pickle


def batch_iterator(dset, batch_size, drop_last=False, shuffle=False):
	batch = []
	for item in dset:
		batch.append(item)
		if len(batch) == batch_size:
			yield batch
			batch = []
	if len(batch) > 0 and not drop_last:
		yield batch


class RetrieverTrainer(Trainer):
	def __init__(
		self,
		model: Union[PreTrainedModel, nn.Module],
		args: RetrievalQATrainingArguments,
		eval_config: EvaluatorConfig,
		eval_wrapper_class: Type[RetrievalModel],
		eval_search_kwargs: Dict[str, Any],
		data_collator: Optional[DataCollator] = None,
		train_dataset: Optional[Dataset] = None,
		eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		model_init: Optional[Callable[[], PreTrainedModel]] = None,
		compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		callbacks: Optional[List[TrainerCallback]] = None,
		optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  # type: ignore
		preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
	):
		super().__init__(
			model=model,
			args=args,
			data_collator=data_collator,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			tokenizer=tokenizer,
			model_init=model_init,
			compute_metrics=compute_metrics,
			callbacks=callbacks,
			optimizers=optimizers,
			preprocess_logits_for_metrics=preprocess_logits_for_metrics
		)
		self.evaluator_config = eval_config
		self.eval_wrapper_class = eval_wrapper_class
		self.eval_search_kwargs = eval_search_kwargs
		_supported_encoders = (BertModel, BertForMaskedLM)
		if not isinstance(self.model, _supported_encoders):
			raise NotImplementedError(f"Model architecture is not supported.")
		return
	
	def mean_pooling(self, token_embeddings, mask):
		token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
		sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
		return sentence_embeddings
	
	def compute_embedding(self, encoded_inputs, outputs):
		if isinstance(self.model, BertModel):
			embedding = self.mean_pooling(outputs.last_hidden_state, encoded_inputs['attention_mask'])
		elif isinstance(self.model, BertForMaskedLM):
			embedding = outputs.hidden_states[-1][:, 0, :]  # take the CLS token
		else:
			raise NotImplementedError
		return embedding
	
	def embed_document_batch(self, model, batch, batch_size=8):
		b = batch_iterator(batch, batch_size=batch_size, shuffle=False)
		embeddings = []
		for bb in b:
			encoded_inputs = self.tokenizer(
				bb, return_tensors="pt",
				padding='longest'
			)
			for k, v in encoded_inputs.items():
				encoded_inputs[k] = v.to(model.device)
			outputs = model(**encoded_inputs, output_hidden_states=True)
			# [len(texts), 768)]
			embedding = self.compute_embedding(encoded_inputs, outputs)
			embeddings.append(embedding)
		embeddings = torch.concat(embeddings, dim=0)
		return embeddings

	def compute_loss(self, model, inputs, return_outputs=False):
		if model.additional_training_args.contrastive_loss == 'inbatch_contrastive':
			loss = self._inbatch_contrastive_w_hardneg(model, inputs, return_outputs)
		elif model.additional_training_args.contrastive_loss == 'constructed_contrastive':
			loss = self._constructed_contrastive(model, inputs, return_outputs)
		else:
			raise NotImplementedError
		return loss
	
	def _constructed_contrastive(self, model, inputs, return_outputs=False):
		temperature = model.additional_training_args.temperature

		loss = torch.tensor(0., device=model.device)
		for d in inputs:
			query = d['question']
			gold_doc = d['pos_doc']
			negatives = d['hard_neg_docs'] + d['easy_neg_docs']

			all_texts = [query, gold_doc] + negatives
			# batch it here
			embeddings = self.embed_document_batch(model, all_texts, batch_size=4)

			query_embedding = torch.unsqueeze(embeddings[0], dim=0)
			key_embedding = embeddings[1:]

			# similiarty is doc product
			labels = torch.zeros(1, dtype=torch.long, device=model.device)
			scores = torch.einsum('id, jd->ij', query_embedding / temperature, key_embedding)
			loss += torch.nn.functional.cross_entropy(scores, labels)
		loss /= len(inputs)
		return loss
	
	def _inbatch_contrastive_w_hardneg(self, model, inputs, return_outputs=False):
		hard_neg_ratio = model.additional_training_args.hard_neg_ratio
		temperature = model.additional_training_args.temperature

		bsz = len(inputs)
		num_hard_negs = int(hard_neg_ratio * bsz)

		# collect data
		gathered_query = []
		gathered_gold_doc = []
		gathered_hard_neg_doc = []
		gathered_easy_neg_doc = []
		for d in inputs:
			query = d['question']
			gold_doc = d['pos_doc']
			hard_neg_doc = d['hard_neg_docs']
			easy_neg_doc = gold_doc  # in-batch contrastive
			gathered_query.append(query)
			gathered_gold_doc.append(gold_doc)
			gathered_hard_neg_doc.extend(hard_neg_doc)
			gathered_easy_neg_doc.append(easy_neg_doc)

		included_hard_negs = random.sample(gathered_hard_neg_doc, num_hard_negs)
		all_docs = gathered_gold_doc + included_hard_negs + gathered_easy_neg_doc
		labels = torch.arange(0, bsz, dtype=torch.long, device=model.device)

		query_embeddings = self.embed_document_batch(model, gathered_query, batch_size=4)
		key_embeddings = self.embed_document_batch(model, all_docs, batch_size=4)

		score = torch.einsum('id, jd->ij', query_embeddings / temperature, key_embeddings)
		loss = torch.nn.functional.cross_entropy(score, labels)
		return loss
	
	def prediction_step(
		self,
		model: torch.nn.Module,
		inputs: Dict[str, Union[torch.Tensor, Any]],
		prediction_loss_only: bool,
		ignore_keys: Optional[List[str]] = None,
	) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		with torch.no_grad():
			loss = self.compute_loss(model, inputs, return_outputs=False)
		return loss, None, None
	
	def _load_all_docs(self, document_path) -> List[Document]:
		with open(document_path, "rb") as f:
			documents = pickle.load(f)
		return documents
	
	def _load_eval_data(self, eval_data_path) -> List[Dict]:
		with open(eval_data_path, "rb") as f:
			eval_data = pickle.load(f)
		flattened_eval_data = []
		for d in eval_data:
			for q in d['questions']:
				new_data = deepcopy(d)
				new_data['question'] = q
				flattened_eval_data.append(new_data)
		return flattened_eval_data
	
	def evaluation_loop(
		self,
		dataloader,
		description: str,
		prediction_loss_only = None,
		ignore_keys = None,
		metric_key_prefix: str = "eval",
	) -> EvalLoopOutput:
		is_test_only_mode = (not self.args.do_train) and self.args.do_eval
		if is_test_only_mode:
			model = self.model
			output = EvalLoopOutput(predictions=[], label_ids=None, metrics={}, num_samples=len(dataloader.dataset))
		else:
			model = self._wrap_model(self.model, training=False, dataloader=dataloader)
			output = super().evaluation_loop(
				dataloader,
				description=description,
				prediction_loss_only=prediction_loss_only,
				ignore_keys=ignore_keys,
				metric_key_prefix=metric_key_prefix
			)

		loaded_documents = self._load_all_docs(self.args.documents_path)
		loaded_eval_data = self._load_eval_data(self.args.eval_data_path)

		wrapped_model: RetrievalModel
		if is_test_only_mode and hasattr(model, 'wrapped_model'):
			# we may have preloaded the index
			wrapped_model = model.wrapped_model
		else:
			wrapped_model = self.eval_wrapper_class(model, tokenizer=self.tokenizer, search_kwargs=self.eval_search_kwargs)
		indices = wrapped_model.build_index(loaded_documents, format_str=self.args.retriever_format)
		
		evaluator = RetrieverEvaluator(
			config=self.evaluator_config,
			test_data=loaded_eval_data,
			documents=loaded_documents,
			indexes=indices
		)
		performance, predictions = evaluator.evaluate(wrapped_model, prefix=metric_key_prefix)
		output.metrics.update(performance)

		if self.args.write_predictions:
			save_name = f'step-{self.state.global_step}-predictions.pkl'
			save_path = os.path.join(self.args.output_dir, save_name)
			with open(save_path, 'wb') as f:
				pickle.dump(predictions, f)
		return output