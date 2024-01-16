from copy import deepcopy
from transformers import Trainer, BertModel, BertForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
from typing import Optional, List, Union, Dict, Any, Tuple, Type, Callable
from open_rqa.schema.document import Document
from open_rqa.retrievers.faiss_retriever import FaissRetriever
from open_rqa.evaluation.evaluator import RetrieverEvaluator, EvaluatorConfig
from open_rqa.trainers.retriever.arguments import DataArguments, ContrasitiveTrainingArgs, RetrievalQATrainingArguments
from open_rqa.trainers.retriever.embeddings import embed_document_batch, LocalEmbeddings
import torch
import torch.nn as nn
import random
import os
import pickle
import jsonlines


class RetrieverTrainer(Trainer):
	def __init__(
		self,
		model: Union[PreTrainedModel, nn.Module],
		training_args: RetrievalQATrainingArguments,
		data_args: DataArguments,
		contrastive_args: ContrasitiveTrainingArgs,
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
		self.contrastive_args = contrastive_args
		self.evaluator_config = eval_config
		self.eval_search_kwargs = eval_search_kwargs
		_supported_encoders = (BertModel, BertForMaskedLM)
		if not isinstance(self.model, _supported_encoders):
			raise NotImplementedError(f"Model architecture is not supported.")
		return

	def compute_loss(self, model, inputs, return_outputs=False):
		if self.contrastive_args.contrastive_loss == 'inbatch_contrastive':
			loss = self._inbatch_contrastive_w_hardneg(model, inputs, return_outputs)
		else:
			raise NotImplementedError
		return loss
	
	def _inbatch_contrastive_w_hardneg(self, model, inputs, return_outputs=False):
		hard_neg_ratio = self.contrastive_args.hard_neg_ratio
		temperature = self.contrastive_args.temperature

		bsz = len(inputs)
		num_hard_negs = int(hard_neg_ratio * bsz)

		# collect data
		gathered_query = []
		gathered_gold_doc = []
		gathered_hard_neg_doc = []
		for d in inputs:
			query = d['question']
			gold_doc = d['gold_docs'][0]  # there is only one gold_doc in the list
			hard_neg_doc = d['hard_neg_docs']
			gathered_query.append(query)
			gathered_gold_doc.append(gold_doc)
			gathered_hard_neg_doc.extend(hard_neg_doc)

		gathered_hard_neg_doc = [hard_neg for hard_neg in gathered_hard_neg_doc if hard_neg not in gathered_gold_doc]
		included_hard_negs = random.sample(gathered_hard_neg_doc, num_hard_negs)
		all_docs = gathered_gold_doc + included_hard_negs
		labels = torch.arange(0, bsz, dtype=torch.long, device=model.device)

		query_embeddings = embed_document_batch(self.tokenizer, model, gathered_query, batch_size=bsz)
		key_embeddings = embed_document_batch(self.tokenizer, model, all_docs, batch_size=bsz)
        
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
		with jsonlines.open(eval_data_path) as fread:
			eval_data = list(fread)
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