from dataclasses import dataclass, fields, field
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm
from open_rqa.schema.document import Document
from open_rqa.evaluation.metrics import METRICS, MonitoringMetric, RunningMetic
from open_rqa.retrievers.base import RetrievalOutput
from open_rqa.schema.dialogue import RQAOutput
from open_rqa.pipelines.retrieval_qa import RQAPipeline
from collections import defaultdict

import math


@dataclass
class EvaluatorConfig:
    """controls what metrics to compute during evaluation, as well as other eval related model configs
    """
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
    gen_bleu: bool = field(
        default=False,
        metadata={"help": "Whether to compute BLEU for generation"},
    )
    gen_latency: bool = field(
        default=True,
        metadata={"help": "Whether to compute latency for generation"},
    )
    gen_answer_stats: bool = field(
        default=True,
        metadata={"help": "Whether to compute answer stats for generation"},
    )
    gen_gpt4eval: bool = field(
        default=False,
        metadata={"help": "Whether to prompt GPT-4 as judge for evaluation"},
    )
    e2e_latency: bool = field(
        default=True,
        metadata={"help": "Whether to compute latency for end-to-end"},
    )
    ## eval related model configs
    assistant_prefix: str = field(
        default="ASSISTANT",
        metadata={"help": "Prefix for assistant in a conversation"},
    )
    user_prefix: str = field(
        default="USER",
        metadata={"help": "Prefix for user in a conversation"},
    )
    sep_user: str = field(
        default=" ",
        metadata={"help": "Token right after user finished his/her turn"},
    )
    sep_sys: str = field(
        default="</s>",
        metadata={"help": "Token right after assistant finished his/her turn"},
    )


class Evaluator(ABC):
    def __init__(
        self,
        config: EvaluatorConfig,
        test_data: List[Dict],
        documents = None
    ):
        self.config = config
        self.test_data = test_data
        self.documents = documents
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
        documents = None):
        super().__init__(config, test_data, documents)
        self.retr_metrics = self.init_metrics(metric_type="retr")
        self.e2e_metrics = self.init_metrics(metric_type="e2e")
        return
    
    def init_metrics(self, metric_type):
        metrics = []
        for f in fields(self.config):
            if f.name.startswith(f"{metric_type}_") and getattr(self.config, f.name):
                metric_name = f.name.replace(f"{metric_type}_", "")
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

    def evaluate(self, wrapped_model, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
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
            retr_output: RetrievalOutput = wrapped_model.retrieve(batch["question"])
            retrieved_docs: List[List[Document]] = retr_output.batch_source_documents
            # Recall@1
            retrieved_docs = [[docs[0]] for docs in retrieved_docs] # comment out if want to get result for Recall@4

            gold_docs = []
            for gdoc in batch["gold_docs"]:
                gold_docs.append([Document.from_dict(doc) for doc in gdoc])

            for metric in self.retr_metrics:
                if isinstance(metric, MonitoringMetric):
                    metric.stop(batch['__len__'])
                else:
                    metric.update(retrieved_docs, gold_docs)
            
            for idx in range(batch['__len__']):
                question = batch["question"][idx]
                gold_doc = gold_docs[idx]
                retrieved_doc = retrieved_docs[idx]
                predictions.append({
                    "question": question,
                    "gold_docs": [doc.to_dict() for doc in gold_doc],
                    "retrieved_docs": [doc.to_dict() for doc in retrieved_doc],
                })

        for metric in self.e2e_metrics:
            metric.stop(num_samples_seen)

        performance = self.compute_performance(prefix)
        return performance, predictions


class E2EEvaluator(Evaluator):
    """evaluates end-to-end performance (QA accuracy, and retrieval accurarcy) of a retrieval-based QA model

    Args:
        Evaluator (_type_): _description_
    """
    def __init__(self,
        config: EvaluatorConfig,
        test_data: List[Dict],
    ):
        super().__init__(config, test_data, None)
        self.retr_metrics = self.init_metrics(metric_type="retr")
        self.gen_metrics = self.init_metrics(metric_type="gen")
        self.e2e_metrics = self.init_metrics(metric_type="e2e")
        return
    
    def init_metrics(self, metric_type):
        metrics = []
        for f in fields(self.config):
            if f.name.startswith(f"{metric_type}_") and getattr(self.config, f.name):
                metric_name = f.name.replace(f"{metric_type}_", "")
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

    def evaluate(self, wrapped_model: RQAPipeline, prefix='eval') -> Tuple[Dict[str, Any], List[Dict]]:
        test_data_iterator = self._get_data_iterator()
        self.reset_all_metrics()

        performance = {}
        predictions = []

        # e2e and score
        for metric in self.e2e_metrics:
            metric.start()
        
        num_samples_seen = 0
        num_batches = math.ceil(len(self.test_data) / self.config.batch_size)
        for batch in tqdm(test_data_iterator, desc="Evaluating Performance", total=num_batches):
            num_samples_seen += batch['__len__']
            questions: List[str] = batch["question"]
            gold_docs: List[List[Document]] = batch["gold_docs"]
            gold_answers: List[str] = batch["gold_answer"]
            # e2e qa and score
            for metric in self.gen_metrics:
                if isinstance(metric, MonitoringMetric):
                    metric.start()
            
            gen_outputs: RQAOutput = wrapped_model.qa(
                batch_questions=questions,
                batch_dialogue_session=batch["dialogue_session"],
            )
            retrieved_docs: List[List[Document]] = gen_outputs.batch_source_documents
            generated_answers: List[str] = gen_outputs.batch_answers

            for metric in self.retr_metrics:
                # we do not use monitoring metrics for retrieval
                if isinstance(metric, RunningMetic):
                    metric.update(retrieved_docs, gold_docs)
            for metric in self.gen_metrics:
                if isinstance(metric, MonitoringMetric):
                    metric.stop(batch['__len__'])
                else:
                    metric.update(questions, generated_answers, gold_answers, retrieved_docs, gold_docs)
            
            # flatten and make it savable with jsonlines
            for idx in range(batch['__len__']):
                question = questions[idx]
                gold_doc = gold_docs[idx]
                retrieved_doc = retrieved_docs[idx]
                gold_answer = gold_answers[idx]
                generated_answer = generated_answers[idx]
                predictions.append({
                    "question": question,
                    "gold_docs": [doc.to_dict() for doc in gold_doc],
                    "retrieved_docs": [doc.to_dict() for doc in retrieved_doc],
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                })

        for metric in self.e2e_metrics:
            metric.stop(num_samples_seen)

        performance = self.compute_performance(prefix)
        return performance, predictions