from copy import deepcopy
from transformers import Trainer, BertModel, BertForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from typing import Optional, List, Union, Dict, Any, Tuple, Type, Callable
from open_rqa.schema.document import Document
from open_rqa.retrievers.faiss_retriever import FaissRetriever
from open_rqa.evaluation.evaluator import RetrieverEvaluator, EvaluatorConfig
from open_rqa.trainers.retriever.arguments import DataArguments, FidTrainingArgs, RetrievalQATrainingArguments
from open_rqa.trainers.retriever.embeddings import embed_document_batch, LocalEmbeddings
import torch
import torch.nn as nn
import random
import os
import pickle
import jsonlines
import numpy as np
import json

class FidRetrieverTrainer(Trainer):
	def __init__(
		self,
		model: Union[PreTrainedModel, nn.Module],
		training_args: RetrievalQATrainingArguments,
		data_args: DataArguments,
		fid_args: FidTrainingArgs,
		eval_config: EvaluatorConfig,
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
			args=training_args,
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
		self.data_args = data_args
		self.fid_args = fid_args
		self.evaluator_config = eval_config
		self.eval_search_kwargs = eval_search_kwargs
		self.loss_fct = torch.nn.KLDivLoss()
		if self.fid_args.projection:
			self.proj = nn.Linear(
				self.model.config.hidden_size,
				self.fid_args.indexing_dimension
			).to("cuda:0")
			self.norm = nn.LayerNorm(self.fid_args.indexing_dimension).to("cuda:0")
		_supported_encoders = (BertModel, BertForMaskedLM)
		if not isinstance(self.model, _supported_encoders):
			raise NotImplementedError(f"Model architecture is not supported.")
		return
	
	def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
		text_output = self.model(
			input_ids=text_ids,
			attention_mask=text_mask if apply_mask else None
		)
		if type(text_output) is not tuple:
			text_output.to_tuple()
		text_output = text_output[0]
		if self.fid_args.projection:
			text_output = self.proj(text_output)
			text_output = self.norm(text_output)

		if extract_cls:
			text_output = text_output[:, 0]
		else:
			if apply_mask:
				text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
				text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
			else:
				text_output = torch.mean(text_output, dim=1)
		return text_output

	def kldivloss(self, score, gold_score):
		gold_score = torch.softmax(gold_score/self.fid_args.reader_temperature, dim=-1)
		score = torch.nn.functional.log_softmax(score, dim=-1)
		return self.loss_fct(score, gold_score)

	def compute_loss(self, model, inputs, return_outputs=False):

		(idx, question_ids, question_mask, passage_ids, passage_mask, gold_score) = inputs

		question_output = self.embed_text(
			text_ids=question_ids,
			text_mask=question_mask,
			apply_mask=self.fid_args.apply_question_mask,
			extract_cls=self.fid_args.extract_cls,
		)
		bsz, n_passages, plen = passage_ids.size()
		passage_ids = passage_ids.view(bsz * n_passages, plen)
		passage_mask = passage_mask.view(bsz * n_passages, plen)
		passage_output = self.embed_text(
			text_ids=passage_ids,
			text_mask=passage_mask,
			apply_mask=self.fid_args.apply_passage_mask,
			extract_cls=self.fid_args.extract_cls,
		)

		score = torch.einsum(
			'bd,bid->bi',
			question_output,
			passage_output.view(bsz, n_passages, -1)
		)
		score = score / np.sqrt(question_output.size(-1))
		if gold_score is not None:
			loss = self.kldivloss(score, gold_score)
		else:
			loss = None
		
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
	
	def wrap_model_for_eval(
		self,
		documents,
		embeddings,
		index_path,
	) -> FaissRetriever:
		wrapped_model = FaissRetriever(
			documents,
			embeddings=embeddings,
			index_path=index_path
		)
		return wrapped_model
	
	def _load_all_docs(self, document_path) -> List[Document]:
		with open(document_path, "rb") as f:
			documents = pickle.load(f)
		return documents
	
	def _load_eval_data(self, eval_data_path) -> List[Dict]:
		eval_data = json.load(open(eval_data_path, "r"))
		return eval_data
	
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

		loaded_documents = self._load_all_docs(self.data_args.full_dataset_file_path)
		loaded_eval_data = self._load_eval_data(self.data_args.eval_file)

		wrapped_model = self.wrap_model_for_eval(
			loaded_documents, 
			LocalEmbeddings(model, self.tokenizer), 
			index_path=os.path.join(self.args.output_dir, f"step-{self.state.global_step}-index")
		)
		
		evaluator = RetrieverEvaluator(
			config=self.evaluator_config,
			test_data=loaded_eval_data,
			documents=loaded_documents
		)
		performance, predictions = evaluator.evaluate(wrapped_model, prefix=metric_key_prefix)
		output.metrics.update(performance)

		if self.args.write_predictions:
			save_name = f'step-{self.state.global_step}-predictions.jsonl'
			save_path = os.path.join(self.args.output_dir, save_name)
			with jsonlines.open(save_path, 'w') as fwrite:
				fwrite.write_all(predictions)
		return output

	def _save(self, output_dir: Optional[str] = None, state_dict=None):
		TRAINING_ARGS_NAME = "training_args.bin"
		# If we are executing this function, we are the process zero, so we don't check for that.
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		os.makedirs(output_dir, exist_ok=True)
		
		self.model.save_pretrained(
			output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
		)

		if self.tokenizer is not None:
			self.tokenizer.save_pretrained(output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

		# save to sentence transformers
		word_embedding_model = models.Transformer(output_dir)
		pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.args.pooling_type)
		model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
		model.save(output_dir)