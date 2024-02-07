from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class LoggerArguments:
    """
    Arguments pertaining to using wandb for logging
    """

    run_group: str = field(
        default="debug",
        metadata={"help": "wandb run group"}
    )
    run_project: str = field(
	    default="localRQA",
	    metadata={"help": "wandb run project"}
	)
    run_entity: str = field(
	    default="localRQA",
	    metadata={"help": "wandb run entity"}
	)
    

@dataclass
class ModelArguments:
	model_name_or_path: str = field(
		default="facebook/contriever-msmarco",
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
	)


@dataclass
class FidTrainingArgs:
	reader_model_path: str = field(
		default="google/flan-t5-xl",
		metadata={"help": "Reader model path to get the cross attention score"},
	)
	with_score: bool = field(
		default=False,
		metadata={"help": "Whether the train_file and eval_file dataset have already got crossattention score"}
	)
	text_maxlength: int = field(
		default=512,
		metadata={"help": "maximum number of tokens in text segments (question+passage)"},
	)
	n_context: int = field(
		default=50,
		metadata={"help": "num of candidates passages for each question"},
	)
	apply_question_mask: bool = field(
		default=True
	)
	apply_passage_mask: bool = field(
		default=True
	)
	extract_cls: bool = field(
		default=False
	)
	projection: bool = field(
		default=False
	)
	reader_temperature: float = field(
		default=0.1,
		metadata={"help": "Temperature for lm likelihood score"},
	)
	reader_batch_size: int = field(
		default=4,
		metadata={"help": "The batch size for reader to calculate cross attention score"},
	)
	indexing_dimension: int = field(
		default=768,
		metadata={"help": "token embedding dimension"},
	)


@dataclass
class ReplugTrainingArgs:
	lm_model_path: str = field(
		default="stabilityai/stablelm-zephyr-3b",
		metadata={"help": "lm model path to compute LM likelihood"},
	)
	text_maxlength: int = field(
		default=512,
		metadata={"help": "maximum number of tokens in text segments (question+passage)"},
	)
	lm_temperature: float = field(
		default=0.1,
		metadata={"help": "Temperature for lm likelihood score"},
	)
	retrieve_temperature: float = field(
		default=0.1,
		metadata={"help": "Temperature for retrieval similarity score"},
	)
	num_docs: int = field(
		default=20,
		metadata={"help": "number of documents retrieved for each question in training"},
	)
	refresh_step: int = field(
		default=10,
		metadata={"help": "The document index refresh steps"}
	)


@dataclass
class DataArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	train_file: str = field(
		default="data/training/databricks_new/train_w_qa.jsonl",
		metadata={"help": "Path for cached train dataset"},
	)
	eval_file: str = field(
		default='data/training/databricks_new/eval_w_qa.jsonl',
		metadata={"help": "Path for cached eval dataset"},
	)
	test_file: str = field(
		default='data/training/databricks_new/test_w_qa.jsonl',
		metadata={"help": "Path for cached test dataset"},
	)
	full_dataset_file_path: str = field(
		default='data/database/databricks/databricks_400.pkl',
		metadata={"help": "Path for cached full dataset file"},
	)


@dataclass
class ContrasitiveTrainingArgs:
	hard_neg_ratio: float = field(
		default=0.05,
		metadata={"help": "Ratio of hard negatives to sample from the batch"},
	)
	contrastive_loss: str = field(  # No Need, since this is the only option for now
		default='inbatch_contrastive',
		metadata={"help": "Type of contrastive loss to use"},
	)
	temperature: float = field(
		default=0.05,
		metadata={"help": "Temperature for contrastive loss"},
	)


@dataclass
class RetrievalQATrainingArguments(TrainingArguments):

	"""
	Arguments overriding some default TrainingArguments
	"""
	output_dir: str = field(
		 default="result/model_checkpoints/contriever/contriever_contrastive_ft",
		 metadata={"help": "The final output dir to save the model and prediction result"}
	)
	do_train: bool = field(
		default=True,
		metadata={"help": "Whether to run training."}
	)
	remove_unused_columns: bool = field(
		default=False, 
		metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
	)
	do_eval: bool = field(
		default=True,
		metadata={"help": "Whether to run eval on the dev set."}
	)
	learning_rate: float = field(
		default=1e-5,
		metadata={"help": "The peak learning rate for the scheduler."}
	)
	weight_decay: float = field(
		default=0.01,
		metadata={"help": "Weight decay to apply to the optimizer."}
	)
	max_steps: int = field(
		 default=200,
		 metadata={"help": "The total number of training steps to perform."}
	)
	per_device_train_batch_size: int = field(
		default=256,
		metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
	)
	per_device_eval_batch_size: int = field(
		default=8,
		metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
	)
	warmup_ratio: float = field(
		default=0.1,
		metadata={"help": "Ratio of warmup steps to total steps."}
	)
	gradient_checkpointing: bool = field(
		default=True,
		metadata={"help": "Whether use gradient checkpointing"},
	)
	lr_scheduler_type: str = field(
		default="cosine",
		metadata={"help": "The scheduler type to use."}
	)
	logging_steps: int = field(
		default=10,
		metadata={"help": "Log every X updates steps."}
	)
	eval_steps: int = field(
		default=100,
		metadata={"help": "Run an evaluation every X steps."}
	)
	save_steps: int = field(
		default=100,
		metadata={"help": "Save checkpoint every X steps."}
	)
	report_to: str = field(
		default="wandb",
		metadata={"help": "Report to wandb or not"}
	)
	evaluation_strategy: str = field(
		default="steps",
		metadata={"help": "Evaluation strategy to adopt during training."}
	)
	metric_for_best_model: str = field(
		default="eval_loss",
		metadata={"help": "The metric to use to compare two different models."}
	)
	save_strategy: str = field(
		default="steps",
		metadata={"help": "Save strategy to adopt during training."}
	)
	save_total_limit: int = field(
		default=1,
		metadata={"help": "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir."}
	)
	seed: int = field(
		default=42,
	)
	write_predictions: bool = field(
		default=True,
		metadata={"help": "Whether to save the predictions to a file"},
	)
	pooling_type: str = field(
		default="mean",
		metadata={"help": "pooling method for embedding, choose from [mean, cls]"}
	)