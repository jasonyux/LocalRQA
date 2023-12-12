import time
import numpy as np
import evaluate
from abc import ABC, abstractmethod
from collections import Counter
from functools import partial
from typing import Dict, List
from open_rqa.evaluation.scores import f1, precision
from open_rqa.evaluation.utils import normalize_answer


def mean(l):
	return sum(l) / len(l)


class RunningMetic(ABC):
	@abstractmethod
	def update(self, *args, **kwargs):
		return
	
	@abstractmethod
	def compute(self):
		return
	
	@abstractmethod
	def reset(self):
		return


class MonitoringMetric(ABC):
	@abstractmethod
	def start(self, *args, **kwargs):
		return
	
	@abstractmethod
	def stop(self, *args, **kwargs):
		return
	
	@abstractmethod
	def compute(self):
		return
	
	@abstractmethod
	def reset(self):
		return


def is_same_document(retrieved_doc, gold_doc):
	retr_source = retrieved_doc.metadata["source"]
	gold_source = gold_doc.metadata["source"]
	retr_content = retrieved_doc.page_content
	gold_content = gold_doc.page_content
	if retr_source == gold_source and retr_content == gold_content:
		return True
	return False


def document_similarity(src_doc, target_doc):
	retr_content = src_doc.page_content
	gold_content = target_doc.page_content
	prediction_tokens = normalize_answer(retr_content).split()
	ground_truth_tokens = normalize_answer(gold_content).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())

	cover_percent = num_same / len(ground_truth_tokens)
	return cover_percent


def is_almost_same_document(retrieved_doc, gold_doc):
	cover_percent = document_similarity(retrieved_doc, gold_doc)
	if cover_percent > 0.7:
		return True
	return False


class DocumentAccuracy(RunningMetic):
	def __init__(self, name="document_accuracy"):
		self.name = name
		self.state = {
			"num_seen": 0,
			"num_correct": 0,
			"num_likely_correct": 0,
		}
		self.reset()
		return
	
	def update(self, batch_retrieved_docs, batch_gold_docs):
		bsz = len(batch_retrieved_docs)
		for i in range(bsz):
			retrieved_docs = batch_retrieved_docs[i]
			gold_docs = batch_gold_docs[i]

			# measure if each document retrieved is correct
			for rdoc in retrieved_docs:
				self.state["num_seen"] += 1

				is_found = [is_same_document(rdoc, gdoc) for gdoc in gold_docs]
				if any(is_found):
					self.state["num_correct"] += 1
				
				is_likely_found = [is_almost_same_document(rdoc, gdoc) for gdoc in gold_docs]
				if any(is_likely_found):
					self.state["num_likely_correct"] += 1
		return
	
	def compute(self):
		return {
			"accuracy": self.state["num_correct"] / self.state["num_seen"],
			"likely_accuracy": self.state["num_likely_correct"] / self.state["num_seen"],
		}
	
	def reset(self):
		self.state = {
			"num_seen": 0,
			"num_correct": 0,
			"num_likely_correct": 0,
		}
		return


class DocumentRecall(RunningMetic):
	def __init__(self, name="document_recall"):
		self.name = name
		self.state = {
			"num_seen": 0,
			"num_correct": 0,
			"num_likely_correct": 0,
		}
		self.reset()
		return
	
	def update(self, batch_retrieved_docs, batch_gold_docs):
		bsz = len(batch_retrieved_docs)
		for i in range(bsz):
			self.state["num_seen"] += 1

			retrieved_docs = batch_retrieved_docs[i]
			gold_docs = batch_gold_docs[i]

			# measure if all the documents from the gold set are retrieved
			all_found = []
			all_likely_found = []
			for gdoc in gold_docs:
				is_found = [is_same_document(rdoc, gdoc) for rdoc in retrieved_docs]
				if any(is_found):
					all_found.append(True)
				else:
					all_found.append(False)

				is_likely_found = [is_almost_same_document(rdoc, gdoc) for rdoc in retrieved_docs]
				if any(is_likely_found):
					all_likely_found.append(True)
				else:
					all_likely_found.append(False)
			if all(all_found):
				self.state["num_correct"] += 1
			if all(all_likely_found):
				self.state["num_likely_correct"] += 1
		return
	
	def compute(self):
		return {
			"recall": self.state["num_correct"] / self.state["num_seen"],
			"likely_recall": self.state["num_likely_correct"] / self.state["num_seen"],
		}
	
	def reset(self):
		self.state = {
			"num_seen": 0,
			"num_correct": 0,
			"num_likely_correct": 0,
		}
		return


class F1(RunningMetic):
	def __init__(self, name="f1"):
		self.name = name
		self.f1_metric = partial(f1, normalize_fn=normalize_answer)
		self.state = {
			"f1_ans": [],
			"f1_retr_doc": [],
		}
		self.reset()
		return
	
	def update(self, batch_gen_answers, batch_gold_answers, batch_retrieved_docs, batch_gold_docs):
		bsz = len(batch_retrieved_docs)
		for i in range(bsz):
			gen_ans = batch_gen_answers[i]
			gold_ans = batch_gold_answers[i]
			retr_docs = batch_retrieved_docs[i]
			
			# measure w.r.t gold answer
			f1_ans = self.f1_metric(gen_ans, gold_ans)

			# check how faithful generated answer are to the retrieved docs
			f1_retr_docs = []
			for rdoc in retr_docs:
				retr_d = rdoc.page_content
				score = self.f1_metric(gen_ans, retr_d)
				f1_retr_docs.append(score)
			f1_retr_doc = max(f1_retr_docs)

			self.state["f1_ans"].append(f1_ans)
			self.state["f1_retr_doc"].append(f1_retr_doc)
		return
	
	def compute(self):
		return {
			f'avg_{k}': mean(v) for k, v in self.state.items()
		}
	
	def reset(self):
		self.state = {
			"f1_ans": [],
			"f1_retr_doc": [],
		}
		return
	


class Precision(RunningMetic):
	def __init__(self, name="precision"):
		self.name = name
		self.precision_metric = partial(precision, normalize_fn=normalize_answer)
		self.state = {
			"precision_ans": [],
			"precision_retr_doc": [],
		}
		self.reset()
		return
	
	def update(self, batch_gen_answers, batch_gold_answers, batch_retrieved_docs, batch_gold_docs):
		bsz = len(batch_retrieved_docs)
		for i in range(bsz):
			gen_ans = batch_gen_answers[i]
			gold_ans = batch_gold_answers[i]
			retr_docs = batch_retrieved_docs[i]
			
			# measure w.r.t gold answer
			precision_ans = self.precision_metric(gen_ans, gold_ans)

			# check how faithful generated answer are to the retrieved docs
			precision_retr_docs = []
			for rdoc in retr_docs:
				retr_d = rdoc.page_content
				score = self.precision_metric(gen_ans, retr_d)
				precision_retr_docs.append(score)
			precision_retr_doc = max(precision_retr_docs)

			self.state["precision_ans"].append(precision_ans)
			self.state["precision_retr_doc"].append(precision_retr_doc)
		return
	
	def compute(self):
		return {
			f'avg_{k}': mean(v) for k, v in self.state.items()
		}
	
	def reset(self):
		self.state = {
			"precision_ans": [],
			"precision_retr_doc": [],
		}
		return


class ROUGE(RunningMetic):
	def __init__(self, name="rouge"):
		self.name = name
		self.rouge_metric = evaluate.load("rouge")
		self.state = {
			"rouge1_ans": [],
			"rouge1_retr_doc": [],
			"rouge2_ans": [],
			"rouge2_retr_doc": [],
			"rougeL_ans": [],
			"rougeL_retr_doc": [],
		}
		self.reset()
		return
	
	def update(self, batch_gen_answers, batch_gold_answers, batch_retrieved_docs, batch_gold_docs):
		bsz = len(batch_retrieved_docs)
		for i in range(bsz):
			gen_ans = batch_gen_answers[i]
			gold_ans = batch_gold_answers[i]
			retr_docs = batch_retrieved_docs[i]
			
			# measure w.r.t gold answer
			rouge_ans = self.rouge_metric.compute(predictions=[gen_ans], references=[gold_ans])

			# check how faithful generated answer are to the retrieved docs
			rouge_retr_docs = {"rouge1": [], "rouge2": [], "rougeL": []}
			for rdoc in retr_docs:
				retr_d = rdoc.page_content
				score = self.rouge_metric.compute(predictions=[gen_ans], references=[retr_d])
				rouge_retr_docs["rouge1"].append(score["rouge1"])
				rouge_retr_docs["rouge2"].append(score["rouge2"])
				rouge_retr_docs["rougeL"].append(score["rougeL"])
			rouge_retr_doc = {k: max(v) for k, v in rouge_retr_docs.items()}

			self.state["rouge1_ans"].append(rouge_ans["rouge1"])
			self.state["rouge1_retr_doc"].append(rouge_retr_doc["rouge1"])
			self.state["rouge2_ans"].append(rouge_ans["rouge2"])
			self.state["rouge2_retr_doc"].append(rouge_retr_doc["rouge2"])
			self.state["rougeL_ans"].append(rouge_ans["rougeL"])
			self.state["rougeL_retr_doc"].append(rouge_retr_doc["rougeL"])
		return
	
	def compute(self):
		return {
			f'avg_{k}': mean(v) for k, v in self.state.items()
		}
	
	def reset(self):
		self.state = {
			"rouge1_ans": [],
			"rouge1_retr_doc": [],
			"rouge2_ans": [],
			"rouge2_retr_doc": [],
			"rougeL_ans": [],
			"rougeL_retr_doc": [],
		}
		return


class AnswerStats(RunningMetic):
	def __init__(self, name="answer_stats") -> None:
		self.name = name
		self.state: Dict[str, List] = {
			"num_words": []
		}
		self.reset()
		return
	
	def update(self, batch_gen_answers, batch_gold_answers, batch_retrieved_docs, batch_gold_docs):
		bsz = len(batch_gen_answers)
		for i in range(bsz):
			gen_ans = batch_gen_answers[i]
			num_words = len(gen_ans.split())
			self.state["num_words"].append(num_words)			
		return
	
	def compute(self):
		return {
			'avg_num_words': mean(self.state["num_words"]),
			'total_num_words': sum(self.state["num_words"]),
		}
	
	def reset(self):
		self.state = {
			"num_words": []
		}
		return


class Latency(MonitoringMetric):
	def __init__(self, name="latency"):
		self.name = name
		self.state = {
			"start_time": 0,
			"end_time": -1,
			"num_samples_seen": 0,
			"total_latency": 0,
		}
		self.reset()
		return
	
	def start(self):
		self.state["start_time"] = time.time()
		return
	
	def stop(self, num_samples_seen):
		self.state["end_time"] = time.time()
		self.state["num_samples_seen"] += num_samples_seen
		self.state["total_latency"] += self.state["end_time"] - self.state["start_time"]
		return
	
	def compute(self):
		total_latency = self.state["total_latency"]
		num_samples_seen = self.state["num_samples_seen"]
		return {
			"avg_latency": total_latency / num_samples_seen,
			"total_latency": total_latency,
		}
	
	def reset(self):
		self.state = {
			"start_time": 0,
			"end_time": -1,
			"num_samples_seen": 0,
			"total_latency": 0,
		}
		return


METRICS = {
	"document_accuracy": DocumentAccuracy,
	"document_recall": DocumentRecall,
	'f1': F1,
	'precision': Precision,
	'rouge': ROUGE,
	'answer_stats': AnswerStats,
	"latency": Latency,
}
