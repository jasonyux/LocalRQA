from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class RetrievalQATrainingArguments(TrainingArguments):
    documents_path: str = field(
        default="",
        metadata={"help": "Path to the file which contains List[Document] for building a database index"},
    )
    eval_data_path: str = field(
        default="",
        metadata={
            "help": ("Path to the eval data JSONL file. It needs to contain fields including 'gold_docs' for retriever, "
                    "and 'gold_docs' and 'gold_answers' for E2E QA.")
        },
    )
    write_predictions: bool = field(
        default=True,
        metadata={"help": "Whether to save the predictions to a file"},
    )


@dataclass
class E2EQATrainingArguments(RetrievalQATrainingArguments):
    """
    Arguments overriding some default TrainingArguments
    """
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The peak learning rate for the scheduler."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of warmup steps to total steps."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay to apply to the optimizer."}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report to wandb or not"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training."}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run an evaluation every X steps."}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "The metric to use to compare two different models."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy to adopt during training."}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X steps."}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir."}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load the best model found during training at the end of training."}
    )
    seed: int = field(
        default=42,
    )