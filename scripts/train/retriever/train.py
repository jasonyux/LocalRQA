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
import wandb

from open_rqa.trainers.retriever.arguments import ModelArguments, DataArguments, ContrasitiveTrainingArgs, RetrievalQATrainingArguments
from open_rqa.trainers.retriever.datasets import ContrastiveRetrievalDataset, NoopDataCollator
from open_rqa.trainers.retriever.retriever_trainer import RetrieverTrainer, EvaluatorConfig
from open_rqa.config.retriever_config import SEARCH_CONFIG


if __name__ == "__main__":
	parser = HfArgumentParser(
		dataclass_types=(ModelArguments, DataArguments, ContrasitiveTrainingArgs, RetrievalQATrainingArguments),
		description="QA Retriever training script"
	)
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, logger_args, training_args = parser.parse_json_file(
			json_file=os.path.abspath(sys.argv[1])
		)
	else:
		model_args, data_args, contrastive_args, training_args = parser.parse_args_into_dataclasses()


	with jsonlines.open(data_args.train_file) as fread:
		train_data = list(fread)
	with jsonlines.open(data_args.eval_file) as fread:
		eval_data = list(fread)

	train_dataset = ContrastiveRetrievalDataset(
		train_data, shuffle=True
	)
	eval_dataset = ContrastiveRetrievalDataset(
		eval_data, shuffle=True
	)


	model = AutoModel.from_pretrained(model_args.model_name_or_path)
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

	eval_config = EvaluatorConfig(
		gen_latency = False,
		batch_size = training_args.per_device_eval_batch_size
	)

	search_kwargs = SEARCH_CONFIG[contrastive_args.search_algo]

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
			group='databricks',
			config=all_args,
		)
	##

	trainer = RetrieverTrainer(
		model=model,
		training_args=training_args,
		data_args=data_args,
		contrastive_args=contrastive_args,
		eval_config=eval_config,
		eval_search_kwargs=search_kwargs,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=NoopDataCollator(),
		tokenizer=tokenizer,
	)

	if training_args.do_eval and not training_args.do_train:
		trainer.evaluate()
	else:
		trainer.train()
