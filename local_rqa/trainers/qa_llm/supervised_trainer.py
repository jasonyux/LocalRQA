from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset
from typing import Optional, List, Union, Dict, Any, Tuple, Callable
from local_rqa.retrievers.base import BaseRetriever
from local_rqa.schema.document import Document
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.pipelines.retrieval_qa import SimpleRQA
from local_rqa.evaluation.evaluator import E2EEvaluator, EvaluatorConfig
from local_rqa.trainers.qa_llm.arguments import RetrievalQATrainingArguments
import torch
import torch.nn as nn
import os
import jsonlines


class SupervisedTrainer(Trainer):
    """Equivalent to assuming a fixed retriever. During training input sequence should include the gold doc
    During evaluation, we will include an embedding model of choice and evaluate end-to-end

    Args:
        Trainer (_type_): _description_
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        train_args: RetrievalQATrainingArguments,
        eval_config: EvaluatorConfig,
        eval_retriever: BaseRetriever,
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
        self.eval_retriever = eval_retriever  # already wrapped retriever model mainly for eval
        self.evaluator_config = eval_config
        return

    def compute_loss(self, model, inputs, return_outputs=False):
        # supervised loss
        loss = super().compute_loss(model, inputs, return_outputs)  # pylint: disable=no-member
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
    
    def _load_eval_data(self, eval_data_path) -> List[Dict]:
        with jsonlines.open(eval_data_path) as fread:
            eval_data = list(fread)
        formatted_eval_data = []
        for d in eval_data:
            formatted_eval_data.append({
                'question': d['question'],
                'gold_docs': [Document.from_dict(doc) for doc in d['gold_docs']],
                'gold_answer': d['gold_answer'],
                'dialogue_session': DialogueSession.from_list(d['chat_history']),
            })
        return formatted_eval_data

    def wrap_model_for_eval(
        self,
        retriever: BaseRetriever,
        qa_model,
        tokenizer,
    ) -> SimpleRQA:
        wrapped_model = SimpleRQA.from_huggingface(
            retriever=retriever,
            qa_model=qa_model,
            qa_tokenizer=tokenizer,
            user_prefix=self.evaluator_config.user_prefix,
            assistant_prefix=self.evaluator_config.assistant_prefix,
            sep_user=self.evaluator_config.sep_user,
            sep_sys=self.evaluator_config.sep_sys,
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
            output = super().evaluation_loop(  # pylint: disable=no-member
                dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )

        wrapped_model_for_eval = self.wrap_model_for_eval(
            retriever=self.eval_retriever,
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
            save_name = f'step-{self.state.global_step}-{metric_key_prefix}-predictions.jsonl'
            save_path = os.path.join(self.args.output_dir, save_name)
            with jsonlines.open(save_path, 'w') as fwrite:
                fwrite.write_all(predictions)
        return output