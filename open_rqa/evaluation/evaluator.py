from dataclasses import dataclass, fields, field
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm
from langchain.schema import Document
from open_rqa.evaluation.metrics import METRICS, MonitoringMetric
from open_rqa.retrievers.base import RetrievalOutput
from open_rqa.schema.dialogue import RQAOutput
from open_rqa.train.model.wrappers import RetrievalModel, RetrievalQAModel
from collections import defaultdict

import math


@dataclass
class EvaluatorConfig:
	batch_size: int = field(
		default=4,
		metadata={"help": "Batch size for evaluation"},
	)
	retr_document_accuracy: bool = field(
		default=True,
		metadata={"help": "Whether to compute accurarcy for EACH retrieved document"},
	)
	retr_document_recall: bool = field(
		default=True,
		metadata={"help": "Whether to compute recall on if all gold documents are retrieved"},
	)
	retr_latency: bool = field(
		default=True,
		metadata={"help": "Whether to compute latency for retrieval"},
	)
	gen_f1: bool = field(
		default=True,
		metadata={"help": "Whether to compute F1 for generation"},
	)
	gen_precision: bool = field(
		default=True,
		metadata={"help": "Whether to compute precision for generation"},
	)
	gen_rouge: bool = field(
		default=True,
		metadata={"help": "Whether to compute ROUGE for generation"},
	)
	gen_latency: bool = field(
		default=True,
		metadata={"help": "Whether to compute latency for generation"},
	)
	gen_answer_stats: bool = field(
		default=True,
		metadata={"help": "Whether to compute answer stats for generation"},
	)
	e2e_latency: bool = field(
		default=True,
		metadata={"help": "Whether to compute latency for end-to-end"},
	)


class Evaluator(ABC):
	def __init__(self, 
		config: EvaluatorConfig,
		test_data: List[Dict], 
		documents = None, indexes = None):
		self.config = config
		self.test_data = test_data
		self.documents = documents
		self.indexes = indexes
		return
	
	def _get_data_iterator(self):
		batch = defaultdict(list)
		batch['__len__'] = 0
		for example in self.test_data:
			for k, v in example.items():
				batch[k].append(v)
			batch['__len__'] += 1
			if batch['__len__'] == self.config.batch_size:
				yield batch
				batch = defaultdict(list)
				batch['__len__'] = 0
		if batch['__len__'] > 0:
			yield batch

	def _flatten_performance(self, metric, prefix, metric_type):
		metric_performance = {}
		score = metric.compute()
		if isinstance(score, dict):
			for k, v in score.items():
				metric_performance[f"{prefix}_{metric_type}/{metric.name}/{k}"] = v
		else:
			metric_performance[f"{prefix}_{metric_type}/{metric.name}"] = score
		return metric_performance

	@abstractmethod
	def evaluate(self, wrapped_model, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
		raise NotImplementedError
	

class RetrieverEvaluator(Evaluator):
	def __init__(self, 
		config: EvaluatorConfig,
		test_data: List[Dict], 
		documents = None, indexes = None):
		super().__init__(config, test_data, documents, indexes)
		self.retr_metrics = self.init_metrics(type="retr")
		self.e2e_metrics = self.init_metrics(type="e2e")
		return
	
	def init_metrics(self, type):
		metrics = []
		for field in fields(self.config):
			if field.name.startswith(f"{type}_") and getattr(self.config, field.name):
				metric_name = field.name.replace(f"{type}_", "")
				metric = METRICS[metric_name]()
				metrics.append(metric)
		return metrics
	
	def reset_all_metrics(self):
		for metric in self.retr_metrics:
			metric.reset()
		for metric in self.e2e_metrics:
			metric.reset()
		return

	def compute_performance(self, prefix):
		performance = {}
		for metric in self.retr_metrics:
			# flattent so that trainer can still track nested metrics
			metric_performance = self._flatten_performance(metric, prefix, metric_type="retr")
			performance.update(metric_performance)
		for metric in self.e2e_metrics:
			metric_performance = self._flatten_performance(metric, prefix, metric_type="e2e")
			performance.update(metric_performance)
		return performance

	def evaluate(self, wrapped_model: RetrievalModel, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
		test_data_iterator = self._get_data_iterator()
		self.reset_all_metrics()

		performance = {}
		predictions = []

		# e2e and score
		for metric in self.e2e_metrics:
			metric.start()
		
		num_samples_seen = 0
		num_batches = math.ceil(len(self.test_data) / self.config.batch_size)
		for batch in tqdm(test_data_iterator, total=num_batches, desc="Evaluating"):
			num_samples_seen += batch['__len__']
			# retrieve and score
			for metric in self.retr_metrics:
				if isinstance(metric, MonitoringMetric):
					metric.start()
			retr_output: RetrievalOutput = wrapped_model.retrieve(batch, self.documents, self.indexes)
			retrieved_docs: List[List[Document]] = retr_output.retrieved_docs

			gold_docs = batch["gold_docs"]

			for metric in self.retr_metrics:
				if isinstance(metric, MonitoringMetric):
					metric.stop(batch['__len__'])
				else:
					metric.update(retrieved_docs, gold_docs)
			
			predictions.append({
				"batch": batch,
				"retrieved_docs": retrieved_docs,
				"gold_docs": gold_docs,
			})

		for metric in self.e2e_metrics:
			metric.stop(num_samples_seen)

		performance = self.compute_performance(prefix)
		return performance, predictions


class E2EEvaluator(Evaluator):
	def __init__(self, 
		config: EvaluatorConfig,
		test_data: List[Dict], 
		documents = None, indexes = None):
		super().__init__(config, test_data, documents, indexes)
		self.retr_metrics = self.init_metrics(type="retr")
		self.gen_metrics = self.init_metrics(type="gen")
		self.e2e_metrics = self.init_metrics(type="e2e")
		return
	
	def init_metrics(self, type):
		metrics = []
		for field in fields(self.config):
			if field.name.startswith(f"{type}_") and getattr(self.config, field.name):
				metric_name = field.name.replace(f"{type}_", "")
				metric = METRICS[metric_name]()
				metrics.append(metric)
		return metrics
	
	def reset_all_metrics(self):
		for metric in self.retr_metrics:
			metric.reset()
		for metric in self.gen_metrics:
			metric.reset()
		for metric in self.e2e_metrics:
			metric.reset()
		return

	def compute_performance(self, prefix):
		performance = {}
		for metric in self.retr_metrics:
			# flattent so that trainer can still track nested metrics
			metric_performance = self._flatten_performance(metric, prefix, metric_type="retr")
			performance.update(metric_performance)
		for metric in self.gen_metrics:
			metric_performance = self._flatten_performance(metric, prefix, metric_type="gen")
			performance.update(metric_performance)
		for metric in self.e2e_metrics:
			metric_performance = self._flatten_performance(metric, prefix, metric_type="e2e")
			performance.update(metric_performance)
		return performance

	def evaluate(self, wrapped_model: RetrievalQAModel, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
		test_data_iterator = self._get_data_iterator()
		self.reset_all_metrics()

		performance = {}
		predictions = []

		# e2e and score
		for metric in self.e2e_metrics:
			metric.start()
		
		num_samples_seen = 0
		for batch in test_data_iterator:
			num_samples_seen += batch['__len__']
			# retrieve and score
			for metric in self.retr_metrics:
				if isinstance(metric, MonitoringMetric):
					metric.start()
			retr_output: RetrievalOutput = wrapped_model.retrieve(batch, self.documents, self.indexes)
			retrieved_docs: List[List[Document]] = retr_output.retrieved_docs

			gold_docs = batch["gold_docs"]

			for metric in self.retr_metrics:
				if isinstance(metric, MonitoringMetric):
					metric.stop(batch['__len__'])
				else:
					metric.update(retrieved_docs, gold_docs)
			
			# generate and score
			for metric in self.gen_metrics:
				if isinstance(metric, MonitoringMetric):
					metric.start()
			
			gen_outputs: RQAOutput = wrapped_model.generate_from_docs(batch, retr_output)
			generated_answers: List[str] = gen_outputs.generated_answers

			gold_answers = batch["gold_answer"]
			for metric in self.gen_metrics:
				if isinstance(metric, MonitoringMetric):
					metric.stop(batch['__len__'])
				else:
					metric.update(generated_answers, gold_answers, retrieved_docs, gold_docs)
			
			predictions.append({
				"batch": batch,
				"retrieved_docs": retrieved_docs,
				"gold_docs": gold_docs,
				"generated_answers": generated_answers,
				"gold_answers": gold_answers,
			})

		for metric in self.e2e_metrics:
			metric.stop(num_samples_seen)

		performance = self.compute_performance(prefix)
		return performance, predictions