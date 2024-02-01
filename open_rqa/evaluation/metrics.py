import time
import evaluate
import os
import re
import logging
from abc import ABC, abstractmethod
from collections import Counter
from functools import partial
from typing import Dict, List
from open_rqa.evaluation.scores import f1, precision
from open_rqa.evaluation.utils import normalize_answer
from open_rqa.schema.document import Document
from openai import OpenAI


logger = logging.getLogger(__name__)


def mean(l):
    return sum(l) / len(l)


class RunningMetic(ABC):
    """this metric computes a score for each bach input (using update), and returns an overall score (using compute) at the very end e.g., by averaging

    Args:
        ABC (_type_): _description_
    """
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
    """this metric will start record something (e.g. time elapsed) between calls of start and stop, and return an overall score (using compute) at the very end e.g., by difference

    Args:
        ABC (_type_): _description_
    """
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


def is_almost_same_document(retrieved_doc, gold_doc, threshold=0.7):
    cover_percent = document_similarity(retrieved_doc, gold_doc)
    if cover_percent > threshold:
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
    
    def update(
        self,
        batch_questions: List[str],
        batch_gen_answers: List[str],
        batch_gold_answers: List[str],
        batch_retrieved_docs: List[List[Document]],
        batch_gold_docs: List[List[Document]],
    ):
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
    
    def update(
        self,
        batch_questions: List[str],
        batch_gen_answers: List[str],
        batch_gold_answers: List[str],
        batch_retrieved_docs: List[List[Document]],
        batch_gold_docs: List[List[Document]],
    ):
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
    
    def update(
        self,
        batch_questions: List[str],
        batch_gen_answers: List[str],
        batch_gold_answers: List[str],
        batch_retrieved_docs: List[List[Document]],
        batch_gold_docs: List[List[Document]],
    ):
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


class BLEU(RunningMetic):
    def __init__(self, name="bleu"):
        self.name = name
        self.bleu_metric = evaluate.load("bleu")
        self.state = {
            "bleu_ans": [],
            "bleu_retr_doc": [],
            "brevity_pen_ans": [],
        }
        self.reset()
        return
    
    def update(
        self,
        batch_questions: List[str],
        batch_gen_answers: List[str],
        batch_gold_answers: List[str],
        batch_retrieved_docs: List[List[Document]],
        batch_gold_docs: List[List[Document]],
    ):
        bsz = len(batch_retrieved_docs)
        for i in range(bsz):
            gen_ans = batch_gen_answers[i]
            gold_ans = batch_gold_answers[i]
            retr_docs = batch_retrieved_docs[i]
            
            # measure w.r.t gold answer
            bleu_ans = self.bleu_metric.compute(predictions=[gen_ans], references=[gold_ans])

            # check how faithful generated answer are to the retrieved docs
            bleu_retr_docs = {"bleu": []}
            for rdoc in retr_docs:
                retr_d = rdoc.page_content
                score = self.bleu_metric.compute(predictions=[gen_ans], references=[retr_d])
                bleu_retr_docs["bleu"].append(score["bleu"])
            bleu_retr_doc = {k: max(v) for k, v in bleu_retr_docs.items()}

            self.state["bleu_ans"].append(bleu_ans["bleu"])
            self.state["bleu_retr_doc"].append(bleu_retr_doc["bleu"])
            self.state["brevity_pen_ans"].append(bleu_ans["brevity_penalty"])
        return
    
    def compute(self):
        return {
            f'avg_{k}': mean(v) for k, v in self.state.items()
        }
    
    def reset(self):
        self.state = {
            "bleu_ans": [],
            "bleu_retr_doc": [],
            "brevity_pen_ans": [],
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
    
    def update(
        self,
        batch_questions: List[str],
        batch_gen_answers: List[str],
        batch_gold_answers: List[str],
        batch_retrieved_docs: List[List[Document]],
        batch_gold_docs: List[List[Document]],
    ):
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


GPT_EVAL_ACC_PROMPT = """
Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user question displayed below.
Your evaluation should only consider whether the response answered the user's question and contained the correct information.
Begin your evaluation by providing a SHORT explanation, no more than 5 sentences. Be as objective as possible.
After providing your explanation, you must access whether the response is correct or incorrect by strictly following this format: "[[correctness]]", for example: "Correctness: [[correct]]" or "Correctness: [[incorrect]]".

[Question]
{question}
[Reference Answer]
{reference}
[Reference Passages]
{reference_passages}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

Explanation:
""".strip()


GPT_EVAL_NOANS_ACC_PROMPT = """
Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user question displayed below.
Your evaluation should only consider whether the response answered the user's question and contained the correct information.
Begin your evaluation by providing a SHORT explanation, no more than 5 sentences. Be as objective as possible.
After providing your explanation, you must access whether the response is correct or incorrect by strictly following this format: "[[correctness]]", for example: "Correctness: [[correct]]" or "Correctness: [[incorrect]]".

[Question]
{question}
[Reference Passages]
{reference_passages}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

Explanation:
""".strip()


class GPT4Eval(RunningMetic):
    def __init__(self, name="gpt4eval", use_gold_answer=False):
        self.name = name
        self.use_gold_answer = use_gold_answer

        self.model_name = 'gpt-4-1106-preview'
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY"),
            organization = os.environ.get("OPENAI_ORGANIZATION")
        )
        self.state = {
            "gpt4eval_acc": [],
        }
        self._default_generate_kwargs = {
            "temperature": 0.7,
            "timeout": 30.0,
        }
        self.reset()
        return

    def _generate(self, prompt):
        _gen_kwargs = {
            **self._default_generate_kwargs,
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                **_gen_kwargs
            )
            extracted_message = response.choices[0].message.content
        except Exception as _:
            extracted_message = ''
        return extracted_message

    def judge(self, question, reference, answer, gold_docs, retrieved_docs):
        ### format prompt
        gold_doc = gold_docs[0]
        retr_doc_to_include = None
        for retr_doc in retrieved_docs:
            if not is_almost_same_document(gold_doc, retr_doc):
                retr_doc_to_include = retr_doc
                break
        if retr_doc_to_include is not None:
            passages = [gold_doc, retr_doc_to_include]
        else:
            passages = [gold_doc]
        fmt_passages = "\n".join([p.fmt_content for p in passages])

        if self.use_gold_answer:
            prompt = GPT_EVAL_ACC_PROMPT.format(
                question=question,
                reference=reference,
                reference_passages=fmt_passages,
                answer=answer,
            )
        else:
            prompt = GPT_EVAL_NOANS_ACC_PROMPT.format(
                question=question,
                reference_passages=fmt_passages,
                answer=answer,
            )

        ### generate
        logger.info(f"GPT4Eval Prompt:\n {prompt}")
        extracted_message = self._generate(prompt)
        extracted_correctness = re.findall(r"Correctness: \[\[(.*)\]\]", extracted_message)
        logger.info(f"GPT4Eval Response:\n {extracted_message}")
        logger.info(f"GPT4Eval Correctness: {extracted_correctness}")

        if len(extracted_correctness) == 0:
            return None
        else:
            if extracted_correctness[0].strip().lower() == "correct":
                return True
            else:
                return False
        return
    
    def update(
        self,
        batch_questions: List[str],
        batch_gen_answers: List[str],
        batch_gold_answers: List[str],
        batch_retrieved_docs: List[List[Document]],
        batch_gold_docs: List[List[Document]],
    ):
        bsz = len(batch_retrieved_docs)
        for i in range(bsz):
            question = batch_questions[i]
            gen_ans = batch_gen_answers[i]
            gold_ans = batch_gold_answers[i]
            gold_docs = batch_gold_docs[i]
            retr_docs = batch_retrieved_docs[i]
            
            # measure w.r.t gold answer
            correctness = self.judge(
                question=question,
                reference=gold_ans,
                answer=gen_ans,
                gold_docs=gold_docs,
                retrieved_docs=retr_docs,
            )
            self.state["gpt4eval_acc"].append(correctness)
        return
    
    def compute(self):
        no_none = [x for x in self.state["gpt4eval_acc"] if x is not None]
        return {
            'gpt4eval_acc': mean(no_none)
        }
    
    def reset(self):
        self.state = {
            "gpt4eval_acc": [],
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
    'bleu': BLEU,
    'gpt4eval': GPT4Eval,
    'answer_stats': AnswerStats,
    "latency": Latency,
}
