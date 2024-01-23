from copy import deepcopy
from typing import List, Dict
from open_rqa.schema.document import Document
import torch
import random
import json
import pickle



class NoopDataCollator:
	def __call__(self, features):
		return features


class ContrastiveRetrievalDataset(torch.utils.data.Dataset):
	def __init__(self,
			raw_data: List[Dict],
			start_data_idx=0,
			end_data_idx=None,
			document_fmt_str='title: {title} content: {content}',
			shuffle=False):
		self.start_data_idx = start_data_idx
		self.end_data_idx = end_data_idx
		self.document_fmt_str = document_fmt_str

		self.data = self.prepare_data(raw_data)
		if shuffle:
			# usually the training data files are ALREADY shuffled
			# in the case of few shot experiments, we want to explicitly shuffle the data
			random.seed(42)
			random.shuffle(self.data)
		return
	
	def prepare_data(self, raw_data: List[Dict]):
		flattened_data = []
		for sample in raw_data[self.start_data_idx:self.end_data_idx]:
			questions = sample['questions']
			sample.pop('questions')
			processed_sample = {}
			doc_keys = ['gold_docs', 'hard_neg_docs']
			for k, v in sample.items():
				if k in doc_keys:
					processed_sample[k] = [vv['fmt_content'] for vv in v]
				else:
					processed_sample[k] = v
			for q in questions:
				new_processed_sample = deepcopy(processed_sample)
				new_processed_sample['question'] = q
				flattened_data.append(new_processed_sample)
		return flattened_data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


class Dataset(torch.utils.data.Dataset):
	def __init__(self,
				 data,
				 n_context=None,
				 score_key="score",
				 question_prefix='question:',
				 title_prefix='title:',
				 passage_prefix='context:'):
		self.data = data
		self.n_context = n_context
		self.question_prefix = question_prefix
		self.title_prefix = title_prefix
		self.passage_prefix = passage_prefix
		self.score_key = score_key
		self.sort_data()

	def __len__(self):
		return len(self.data)

	def get_target(self, example):
		if 'target' in example:
			target = example['target']
			return target + ' </s>'
		elif 'answers' in example:
			return random.choice(example['answers']) + ' </s>'
		else:
			return None

	def __getitem__(self, index):
		example = self.data[index]
		question = self.question_prefix + " " + example['question']
		target = self.get_target(example)

		if 'ctxs' in example and self.n_context is not None:
			f = self.title_prefix + " {} " + self.passage_prefix + " {}"
			contexts = example['ctxs'][:self.n_context]
			passages = [f.format(c['title'], c['text']) for c in contexts]
			scores = [float(c[self.score_key]) for c in contexts]
			scores = torch.tensor(scores)
			if len(contexts) == 0:
				contexts = [question]
		else:
			passages, scores = None, None


		return {
			'index' : index,
			'question' : question,
			'target' : target,
			'passages' : passages,
			'scores' : scores
		}

	def sort_data(self):
		if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
			return
		for ex in self.data:
			ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

	def get_example(self, index):
		return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
	passage_ids, passage_masks = [], []
	for k, text_passages in enumerate(batch_text_passages):
		p = tokenizer.batch_encode_plus(
			text_passages,
			padding='max_length',
			max_length=max_length,
			return_tensors='pt',
			truncation=True
		)
		passage_ids.append(p['input_ids'][None])
		passage_masks.append(p['attention_mask'][None])

	passage_ids = torch.cat(passage_ids, dim=0)
	passage_masks = torch.cat(passage_masks, dim=0)
	return passage_ids, passage_masks.bool()

class Collator(object):
	def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
		self.tokenizer = tokenizer
		self.text_maxlength = text_maxlength
		self.answer_maxlength = answer_maxlength

	def __call__(self, batch):
		index = torch.tensor([ex['index'] for ex in batch])

		def append_question(example):
			if example['passages'] is None:
				return [example['question']]
			return [example['question'] + " " + t for t in example['passages']]
		text_passages = [append_question(example) for example in batch]
		passage_ids, passage_masks = encode_passages(text_passages,
													 self.tokenizer,
													 self.text_maxlength)

		return (index, None, None, passage_ids, passage_masks)


def load_data(data_path=None):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


class RetrieverCollator(object):
	def __init__(self, tokenizer, passage_maxlength=512, question_maxlength=512):
		self.tokenizer = tokenizer
		self.passage_maxlength = passage_maxlength
		self.question_maxlength = question_maxlength

	def __call__(self, batch):
		index = torch.tensor([ex['index'] for ex in batch])

		question = [ex['question'] for ex in batch]
		question = self.tokenizer.batch_encode_plus(
			question,
			padding='longest',
			return_tensors="pt",
			max_length=self.question_maxlength,
			truncation=True
		)
		question_ids = question['input_ids']
		question_mask = question['attention_mask'].bool()

		if batch[0]['scores'] is None or batch[0]['passages'] is None:
			return index, question_ids, question_mask, None, None, None

		scores = [ex['scores'] for ex in batch]
		scores = torch.stack(scores, dim=0)

		passages = [ex['passages'] for ex in batch]
		passage_ids, passage_masks = encode_passages(
			passages,
			self.tokenizer,
			self.passage_maxlength
		)

		return (index, question_ids, question_mask, passage_ids, passage_masks, scores)
