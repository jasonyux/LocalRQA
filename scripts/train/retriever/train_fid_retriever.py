import sys
from typing import List, Dict, Type
import pickle
import argparse
import os
import json
import jsonlines

from transformers import (
	AutoTokenizer, AutoModel,
	HfArgumentParser
)
import torch
from torch.utils.data import DataLoader, SequentialSampler
import wandb

from open_rqa.trainers.retriever.arguments import ModelArguments, DataArguments, FidTrainingArgs, RetrievalQATrainingArguments
from open_rqa.qa_llms.fid import FiDT5
from open_rqa.trainers.retriever.datasets import Dataset, Collator, RetrieverCollator, load_data
from open_rqa.trainers.retriever.retriever_fid_trainer import FidRetrieverTrainer, EvaluatorConfig
from open_rqa.config.retriever_config import SEARCH_CONFIG


def evaluate(model, dataset, dataloader, tokenizer):
	loss, curr_loss = 0.0, 0.0
	model.eval()
	if hasattr(model, "module"):
		model = model.module
	model.overwrite_forward_crossattention()
	model.reset_score_storage() 
	with torch.no_grad():
		for i, batch in enumerate(dataloader):
			(idx, _, _, context_ids, context_mask) = batch
			model.reset_score_storage()

			outputs = model.generate(
				input_ids=context_ids.cuda(),
				attention_mask=context_mask.cuda(),
				max_new_tokens=50
			)
			crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

			for k, o in enumerate(outputs):
				ans = tokenizer.decode(o, skip_special_tokens=True)
				example = dataset.data[idx[k]]

				for j in range(context_ids.size(1)):
					example['ctxs'][j]['crossattention_score'] = crossattention_scores[k, j].item()


def get_crossattention_score(model_args, data_file, fid_args, training_args):
	tokenizer = AutoTokenizer.from_pretrained(fid_args.reader_model_path)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	collator_function = Collator(fid_args.text_maxlength, tokenizer)
	eval_examples = load_data(data_file)
	eval_dataset = Dataset(
		eval_examples, 
		fid_args.n_context,
		score_key="score"
	)
	eval_sampler = SequentialSampler(eval_dataset) 
	eval_dataloader = DataLoader(
		eval_dataset, 
		sampler=eval_sampler, 
		batch_size=training_args.per_device_train_batch_size,
		num_workers=20, 
		collate_fn=collator_function
	)
	
	model = FiDT5.from_t5(
		fid_args.reader_model_path,
	)
	model.to("cuda")

	evaluate(model, eval_dataset, eval_dataloader, tokenizer)

	return eval_dataset.data


if __name__ == "__main__":
	parser = HfArgumentParser(
		dataclass_types=(ModelArguments, DataArguments, FidTrainingArgs, RetrievalQATrainingArguments),
		description="QA Retriever training script"
	)
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, logger_args, training_args = parser.parse_json_file(
			json_file=os.path.abspath(sys.argv[1])
		)
	else:
		model_args, data_args, fid_args, training_args = parser.parse_args_into_dataclasses()
	
	# Get cross attention score based on the reader
	train_examples = get_crossattention_score(model_args, data_args.train_file, fid_args, training_args)
	eval_examples = get_crossattention_score(model_args, data_args.eval_file, fid_args, training_args)

		
	#Load data
	model = AutoModel.from_pretrained(model_args.model_name_or_path)
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
	collator_function = RetrieverCollator(
		tokenizer
	)
	train_dataset = Dataset(train_examples, fid_args.n_context, score_key="crossattention_score")
	eval_dataset = Dataset(eval_examples, fid_args.n_context, score_key="crossattention_score")


	eval_config = EvaluatorConfig(
		gen_latency = False,
		batch_size = training_args.per_device_eval_batch_size
	)

	search_kwargs = SEARCH_CONFIG[fid_args.search_algo]

	## temporary code for debug
	if 'wandb' in training_args.report_to:
		all_args = {
			'training_args': training_args.to_dict(),
			'cmd_args': vars(training_args),
		}
		run = wandb.init(
			project='tamarin',
			entity='tamarin',
			name=training_args.output_dir.split("/")[-1] or None,
			group='faire',
			config=all_args,
		)
	##

	trainer = FidRetrieverTrainer(
		model=model,
		training_args=training_args,
		data_args=data_args,
		fid_args=fid_args,
		eval_config=eval_config,
		eval_search_kwargs=search_kwargs,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=collator_function,
		tokenizer=tokenizer,
	)

	if training_args.do_eval and not training_args.do_train:
		trainer.evaluate()
	else:
		trainer.train()
