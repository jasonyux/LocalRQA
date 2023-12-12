from copy import deepcopy
from typing import List, Dict
from open_rqa.schema.document import Document
import torch
import random



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
			for k, v in sample.items():
				if isinstance(v, Document):
					processed_sample[k] = self.document_fmt_str.format(
						title = v.title,
						content = v.content
					)
				elif isinstance(v, list) and isinstance(v[0], Document):
					processed_sample[k] = [
						self.document_fmt_str.format(
							title = vv.title,
							content = vv.content
						) for vv in v
					]
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
