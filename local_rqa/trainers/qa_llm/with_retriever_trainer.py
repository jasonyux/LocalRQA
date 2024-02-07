from copy import deepcopy
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
from typing import Optional, List, Union, Dict, Any, Tuple, Type, Callable
from local_rqa.schema.document import Document
from local_rqa.retrievers.faiss_retriever import FaissRetriever
from local_rqa.pipelines.retrieval_qa import SimpleRQA
from local_rqa.evaluation.evaluator import E2EEvaluator, EvaluatorConfig
from local_rqa.trainers.qa_llm.arguments import RetrievalQATrainingArguments
import torch
import torch.nn as nn
import os
import pickle


def batch_iterator(dset, batch_size, drop_last=False):
    batch = []
    for item in dset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch


class FixedRetrieverTrainer(Trainer):
    def __init__(
        self,
        retriever_model: FaissRetriever,
        model: Union[PreTrainedModel, nn.Module],
        train_args: RetrievalQATrainingArguments,
        eval_config: EvaluatorConfig,
        eval_wrapper_class: Type[SimpleRQA],
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  # type: ignore
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.retriever_model = retriever_model  # already wrapped retriever model
        self.evaluator_config = eval_config
        self.eval_wrapper_class = eval_wrapper_class
        return

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = None
        batch_questions = None
        if self.args.use_gold_docs:
            docs = inputs['gold_docs']
            formatted_inputs = []
        else:
            # 1. retriever fetch documents
            self.retriever_model.retrieve(batch_questions)
            formatted_inputs = []
        return loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return loss, None, None
    
    def _load_all_docs(self, document_path) -> List[Document]:
        with open(document_path, "rb") as f:
            documents = pickle.load(f)
        return documents
    
    def _load_eval_data(self, eval_data_path) -> List[Dict]:
        # TODO: this is bad
        with open(eval_data_path, "rb") as f:
            eval_data = pickle.load(f)
        flattened_eval_data = []
        for d in eval_data:
            for q, a in zip(d['questions'], d['gold_answers']):
                new_data = deepcopy(d)
                new_data['question'] = q
                new_data['gold_answer'] = a
                flattened_eval_data.append(new_data)
        return flattened_eval_data

    def wrap_model_for_eval(
        self,
        retriever: FaissRetriever,
        qa_model,
        tokenizer,
    ) -> SimpleRQA:
        wrapped_model = SimpleRQA.from_huggingface(
            retriever=retriever,
            qa_model=qa_model,
            qa_tokenizer=tokenizer,
            user_prefix="USER",  # doesn't really matter as evaluation during training is single turn
            assistant_prefix="ASSISTANT",
        )
        return wrapped_model
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        is_test_only_mode = (not self.args.do_train) and self.args.do_eval
        if is_test_only_mode:
            model = self.model
            output = EvalLoopOutput(predictions=[], label_ids=None, metrics={}, num_samples=len(dataloader.dataset))
        else:
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            output = super().evaluation_loop(
                dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )

        wrapped_model_for_eval = self.wrap_model_for_eval(
            retriever=self.retriever_model,
            qa_model=model,
            tokenizer=self.tokenizer,
        )

        loaded_eval_data = self._load_eval_data(self.args.eval_data_path)
        evaluator = E2EEvaluator(
            config=self.evaluator_config,
            test_data=loaded_eval_data,
        )
        performance, predictions = evaluator.evaluate(wrapped_model_for_eval, prefix=metric_key_prefix)
        output.metrics.update(performance)

        if self.args.write_predictions:
            save_name = f'step-{self.state.global_step}-{metric_key_prefix}-predictions.pkl'
            save_path = os.path.join(self.args.output_dir, save_name)
            with open(save_path, 'wb') as f:
                pickle.dump(predictions, f)
        return output