import argparse
from dataclasses import dataclass, field
from transformers import TrainingArguments
from open_rqa.config.retriever_config import SEARCH_CONFIG

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        self.parser.add_argument(
		    "--exp_name", type=str, default="contriever_contrastive_ft",
        )
        self.parser.add_argument(
            "--output_base_dir", type=str, default="result/model_checkpoints/contriever",
            help="the final output dir will be output_base_dir/exp_name"
        )
        self.parser.add_argument(
            "--do_train", action='store_true',
        )
        self.parser.add_argument(
            "--do_eval", action='store_true',
        )
        self.parser.add_argument(
            '--model_path', type=str, default='facebook/contriever-msmarco',
        )
        self.parser.add_argument(
            '--model_type', type=str, default='bert',
            choices=['bert', 'bert_mlm'],
        )
        self.parser.add_argument(
            '--documents_path', type=str, 
            default="assets/inventory/contrastive/eval_documents.pkl",  # 12k documents = a subset of the above
            help="Path to the pickle file that stores List[Documents] representing the database to search from during eval/test",
        )
        self.parser.add_argument(
            '--train_file', type=str, default='assets/inventory/contrastive/train.json',
            help="Path to the json file that stores List[Dict] representing the training data. Fields include question, positive_ctxs, negative_ctxs, hard_negative_ctxs.",
        )
        self.parser.add_argument(
            '--eval_file', type=str, default='assets/inventory/contrastive/eval.json',
            help="Path to the json file that stores List[Dict] representing the eval data. Fields include question, positive_ctxs, negative_ctxs, hard_negative_ctxs.",
        )
        self.parser.add_argument(
            '--search_algo', type=str, default='inner_product',
            choices = list(SEARCH_CONFIG.keys()),
        )

        # trainer args
        self.parser.add_argument(
            '--lr', type=float, default=1e-4,
        )
        self.parser.add_argument(
            '--max_steps', type=int, default=400,
        )
        self.parser.add_argument(
            '--per_device_train_batch_size', type=int, default=256,
        )
        self.parser.add_argument(
            '--no_gradient_checkpointing', action='store_true'
        )
        self.parser.add_argument(
            "--logging_steps", type=int, default=10,
        )
        self.parser.add_argument(
            "--eval_steps", type=int, default=50,
        )
        self.parser.add_argument(
            "--save_steps", type=int, default=50,
        )
        
        # additional contrastive training args
        self.parser.add_argument(
            '--hard_neg_ratio', type=float, default=0.05,
        )
        self.parser.add_argument(
            '--contrastive_loss', type=str, default='constructed_contrastive',
            choices=['inbatch_contrastive', 'constructed_contrastive'],
        )
        self.parser.add_argument(
            '--temperature', type=float, default=0.05,
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
    
@dataclass
class ContrasitiveTrainingArgs:
	hard_neg_ratio: float = field(
		default=0.05,
		metadata={"help": "Ratio of hard negatives to sample from the batch"},
	)
	contrastive_loss: str = field(
		default='inbatch_contrastive',
		metadata={"help": "Type of contrastive loss to use"},
	)
	temperature: float = field(
		default=0.05,
		metadata={"help": "Temperature for contrastive loss"},
	)


@dataclass
class RetrievalQATrainingArguments(TrainingArguments):
	documents_path: str = field(
		default="",
		metadata={"help": "Path to the file which contains List[Document] for building a database index"},
	)
	retriever_format: str = field(
		default="title: {title} content: {text}",
		metadata={"help": "Format string for building a database index"},
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