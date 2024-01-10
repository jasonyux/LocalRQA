from typing import List, Dict, Type
import pickle
import argparse
import os
import json

from transformers import BertModel, AutoTokenizer, AutoModel
import wandb

from open_rqa.trainers.retriever.arguments import Options, RetrievalQATrainingArguments, ContrasitiveTrainingArgs
from open_rqa.trainers.retriever.datasets import ContrastiveRetrievalDataset, NoopDataCollator
from open_rqa.trainers.retriever.retriever_trainer import RetrieverTrainer, EvaluatorConfig
from open_rqa.trainers.utils import read_jsonl
from open_rqa.config.retriever_config import SEARCH_CONFIG
from open_rqa.trainers.retriever.model.wrappers import RetrievalModel, RetrieverfromBertModel


if __name__ == "__main__":
	options = Options()
	args = options.parse()

	train_data = read_jsonl(args.train_file)
	eval_data = read_jsonl(args.eval_file)

	train_dataset = ContrastiveRetrievalDataset(
		train_data, shuffle=True
	)
	eval_dataset = ContrastiveRetrievalDataset(
		eval_data, shuffle=True
	)


	if args.model_type == "bert":
		model = AutoModel.from_pretrained(args.model_path)
	else:
		raise NotImplementedError(f"{args.args.model_type} is not supported")
	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

	training_args = RetrievalQATrainingArguments(
		do_train=args.do_train,
		do_eval=args.do_eval,
		output_dir = os.path.join(args.output_base_dir, args.exp_name),
		remove_unused_columns = False,
		gradient_checkpointing = not args.no_gradient_checkpointing,
		learning_rate = args.lr,
		per_device_train_batch_size = args.per_device_train_batch_size,
		per_device_eval_batch_size = args.per_device_eval_batch_size,
		warmup_ratio = 0.1,
		lr_scheduler_type = "cosine",
		max_steps = args.max_steps,
		weight_decay = 0.01,
		# logging
		report_to = "wandb",
		logging_steps = args.logging_steps,
		# evaluation
		documents_path=args.documents_path,
		retriever_format='title: {title} content: {text}',
		eval_data_path=args.eval_file,
		evaluation_strategy = "steps",
		eval_steps = args.eval_steps,
		metric_for_best_model = "eval_retr/document_recall/recall",
		greater_is_better = True,
		save_strategy = "steps",
		save_steps = args.save_steps,
		save_total_limit = 1,
		push_to_hub=False,
	)
	eval_config = EvaluatorConfig(  # type: ignore
		gen_latency = False,
		batch_size = args.eval_bsz
	)
	additional_training_args = ContrasitiveTrainingArgs(
		hard_neg_ratio=args.hard_neg_ratio,
		contrastive_loss=args.contrastive_loss,
		temperature=args.temperature,
	)
	model.additional_training_args = additional_training_args
	search_kwargs = SEARCH_CONFIG[args.search_algo]

	## temporary code for debug
	if 'wandb' in training_args.report_to:
		all_args = {
			'training_args': training_args.to_dict(),
			'cmd_args': vars(args),
		}
		run = wandb.init(
			project='tamarin',
			entity='tamarin',
			name=training_args.output_dir.split("/")[-1] or None,
			group='databricks',
			config=all_args,
		)
	
	wrapper_class: Type[RetrievalModel]
	if args.model_type == "bert":
		wrapper_class = RetrieverfromBertModel
	else:
		raise NotImplementedError

	trainer = RetrieverTrainer(
		model=model,
		args=training_args,
		eval_config=eval_config,
		eval_wrapper_class=wrapper_class,
		eval_search_kwargs=search_kwargs,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=NoopDataCollator(),
		tokenizer=tokenizer,
	)

	if args.do_eval and not args.do_train:
		trainer.evaluate()
	else:
		trainer.train()
