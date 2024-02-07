import os
import pickle
import jsonlines
from typing import Optional, List, Union, Dict, Any, Tuple, Type, Callable
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import Trainer, BertModel, BertForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from transformers import BatchEncoding
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM
)
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
import numpy as np

from local_rqa.schema.document import Document
from local_rqa.retrievers.faiss_retriever import FaissRetriever
from local_rqa.evaluation.evaluator import RetrieverEvaluator, EvaluatorConfig
from local_rqa.trainers.retriever.arguments import DataArguments, ReplugTrainingArgs, RetrievalQATrainingArguments
from local_rqa.trainers.retriever.embeddings import embed_document_batch, LocalEmbeddings
from local_rqa.retrievers.base import BaseRetriever, RetrievalOutput

warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.checkpoint')
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'
# LLM model instruction, different model may use their own specific format
PROMPT = """
<|user|>
This is a chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers using documents from the following context.
Do not mention 'this context' in the assistant's response, since the following context is only visible to the assistant.
----------------
Context:
{formatted_documents}
----------------
{formatted_chat}<|endoftext|>
<|assistant|>
"""


class ReplugRetrieverTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        training_args: RetrievalQATrainingArguments,
        data_args: DataArguments,
        replug_args: ReplugTrainingArgs,
        eval_config: EvaluatorConfig,
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
            args=training_args,
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
        self.data_args = data_args
        self.replug_args = replug_args
        self.evaluator_config = eval_config
        _supported_encoders = (BertModel, BertForMaskedLM)
        if not isinstance(self.model, _supported_encoders):
            raise NotImplementedError(f"Model architecture is not supported.")
        
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            self.replug_args.lm_model_path,
            trust_remote_code=True
            )
        self.lm_device = self.model.device
        self.lm_model.to(self.lm_device)
        self.lm_model.eval()
        self.lm_tokenizer = AutoTokenizer.from_pretrained(
            self.replug_args.lm_model_path,
            return_tensors="pt",
            padding='max_length', max_length=512,
            truncation=True,
            trust_remote_code=True
        )
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.lm_temp = self.replug_args.lm_temperature
        self.retreive_temp = self.replug_args.retrieve_temperature

        return
    
    def instruct(self, question, doc, label_str):
        input_id, target = [], []
        input_str = PROMPT.format(formatted_documents=doc['text'], formatted_chat=question)
        nl_tokens = self.lm_tokenizer('\n').input_ids
        _target = self.lm_tokenizer(label_str).input_ids[:128]
        _input_id = self.lm_tokenizer(input_str).input_ids + nl_tokens

        input_id += _input_id + _target
        target += [IGNORE_TOKEN_ID] * len(_input_id) + _target
        
        assert len(input_id) == len(target)
        if len(input_id) > self.replug_args.text_maxlength:
            input_id = input_id[-self.replug_args.text_maxlength:]
            target = target[-self.replug_args.text_maxlength:]
        else:
            input_id += [self.lm_tokenizer.pad_token_id] * (self.replug_args.text_maxlength - len(input_id))
            target += [IGNORE_TOKEN_ID] * (self.replug_args.text_maxlength - len(target))

        return input_id, target
    
    def get_seq_prob(self, tokenizer, logit_score, targets):
        seq_probs = []
        for sequence, target in zip(logit_score, targets):
            start_idx, end_idx = 0, len(sequence)
            for idx, value in enumerate(target):
                if value != -100:
                    start_idx = idx
                    break
            for idx, value in enumerate(target[start_idx:]):
                if value == -100:
                    end_idx = idx + start_idx
                    break
            logits = sequence[start_idx-1:end_idx-1] # Depends on different generation model, the start_idx/end_idx might start differently
            target_indexes = target[start_idx:end_idx] # Depends on different generation model, the start_idx/end_idx might start differently
            vocab_dist = F.softmax(logits, dim=-1) # [seq_len, vocab_dim]

            target_indexes = target_indexes.to(vocab_dist.device).unsqueeze(1)
            seq_prob = torch.gather(vocab_dist, 1, target_indexes).squeeze(1)
            seq_probs.append(seq_prob.log().mean())
            

        return seq_probs

    def kldivloss(self, retrieve_scores, lm_scores):
        log_retrieve_scores = torch.nn.functional.log_softmax(retrieve_scores/self.retreive_temp, dim=-1)
        lm_scores = torch.tensor(lm_scores).to(self.model.device)
        lm_scores = torch.softmax(lm_scores/self.lm_temp, dim=-1)
        log_retrieve_scores = log_retrieve_scores.float()
        lm_scores = lm_scores.float()
        criterion = torch.nn.KLDivLoss()
        return criterion(log_retrieve_scores, lm_scores)


    def compute_loss(self, model, inputs, return_outputs=False, pred=False):
        # 0. After <refresh_step> steps, refresh the index
        num_docs = self.replug_args.num_docs if not pred else 4
        if not pred:
            refresh_step = self.state.global_step // self.replug_args.refresh_step
            loaded_documents = self._load_all_docs(self.data_args.full_dataset_file_path)
            self.wrapped_model = self.wrap_model_for_eval(
                loaded_documents, 
                LocalEmbeddings(self.model, self.tokenizer, self.args.pooling_type), 
                index_path=os.path.join(self.args.output_dir, f"step-train-{refresh_step}-index"),
                search_type="similarity",
                search_kwargs={'k': num_docs}
            )
        # 1. Retrieve the documents for each question
        batch_query = []
        key_embeddings = []
        for d in inputs:
            batch_query.append(d["question"])
            retr_output: RetrievalOutput = self.wrapped_model.retrieve([d["question"]])
            retrieved_docs: List[List[Document]] = retr_output.batch_source_documents
            retrieved_docs = [doc.to_dict() for doc in retrieved_docs[0]]
            d['ctxs'] = []
            all_docs = []
            add_gold = True
            for idx, doc in enumerate(retrieved_docs):
                if doc['page_content'] == d["gold_docs"][0]["page_content"]:
                    add_gold = False
                    gold_index = idx
                page_content = doc['page_content']
                all_docs.append(page_content)
                ctx = {'text': page_content}
                d['ctxs'].append(ctx)
            # Optional, not necessary to add if don't have gold document
            if add_gold:
                print(RED + "don't have gold in the faiss retrieve" + RESET)
                ctx = {'text': d["gold_docs"][0]["page_content"]}
                d['ctxs'].insert(0, ctx)
                d['ctxs'].pop()
            else:
                print(GREEN + "have gold in the faiss retrieve" + RESET)
                ctx = {'text': d["gold_docs"][0]["page_content"]}
                d['ctxs'].pop(gold_index)
                d['ctxs'].insert(0, ctx)
            batch_embeddings = embed_document_batch(self.tokenizer, model, self.args.pooling_type, all_docs)
            key_embeddings.append(batch_embeddings)
        
        # 2. Get the Retrieval likelihood
        query_embeddings = embed_document_batch(self.tokenizer, model, self.args.pooling_type, batch_query)
        key_embeddings = torch.stack(key_embeddings)
        query_embeddings = query_embeddings.unsqueeze(1).expand(-1, num_docs, -1)
        retrieve_scores = F.cosine_similarity(query_embeddings, key_embeddings, dim=-1)

        # 3. Get the LM likelihood
        lm_scores = []
        for d in inputs:
            question = d['question']
            label_str = d['target']
            gold_answer_probs = []
            for doc in d['ctxs']:
                input_id, target = self.instruct(question, doc, label_str)
                input_ids_tensor = torch.tensor([input_id], dtype=torch.int64)
                attention_mask = input_ids_tensor.ne(self.lm_tokenizer.pad_token_id).long().float()
                inputs_dict = {'input_ids': input_ids_tensor, 'attention_mask': attention_mask}
                inputs_dict = BatchEncoding(inputs_dict, tensor_type='pt').to(self.lm_device)
                targets_tensor = torch.tensor([target], dtype=torch.int).to(self.lm_device)
                targets_tensor = targets_tensor.type(torch.LongTensor)

                with torch.no_grad():	
                    outputs = self.lm_model(**inputs_dict, labels=targets_tensor)
                gold_answer_prob = self.get_seq_prob(self.lm_tokenizer, outputs.logits, targets_tensor)
                gold_answer_probs.extend(gold_answer_prob)

            lm_scores.append(gold_answer_probs)
        loss = self.kldivloss(retrieve_scores, lm_scores)
        return loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False, pred=True)
        return loss, None, None
    
    def wrap_model_for_eval(
        self,
        documents,
        embeddings,
        index_path,
        **kwargs
    ) -> FaissRetriever:
        wrapped_model = FaissRetriever(
            documents,
            embeddings=embeddings,
            index_path=index_path,
            **kwargs
        )
        return wrapped_model
    
    def _load_all_docs(self, document_path) -> List[Document]:
        with open(document_path, "rb") as f:
            documents = pickle.load(f)
        return documents
    
    def _load_eval_data(self, eval_data_path) -> List[Dict]:
        with jsonlines.open(eval_data_path) as fread:
            eval_data = list(fread)
        format_eval_data = []
        for example in eval_data:
            format_eval_data.append({
                'question' : example['question'],
                'gold_docs': example['gold_docs']
                }
            )
        return format_eval_data
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        loaded_documents = self._load_all_docs(self.data_args.full_dataset_file_path)
        self.wrapped_model = self.wrap_model_for_eval(
            loaded_documents, 
            LocalEmbeddings(model, self.tokenizer, self.args.pooling_type), 
            index_path=os.path.join(self.args.output_dir, f"step-eval-{self.state.global_step}-index")
        )

        is_test_only_mode = (not self.args.do_train) and self.args.do_eval
        if is_test_only_mode:
            model = self.model
            output = EvalLoopOutput(predictions=[], label_ids=None, metrics={}, num_samples=len(dataloader.dataset))
        else:
            output = super().evaluation_loop(
                dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )

        loaded_eval_data = self._load_eval_data(self.data_args.eval_file)
        
        evaluator = RetrieverEvaluator(
            config=self.evaluator_config,
            test_data=loaded_eval_data,
            documents=loaded_documents
        )
        performance, predictions = evaluator.evaluate(self.wrapped_model, prefix=metric_key_prefix)
        output.metrics.update(performance)

        if self.args.write_predictions:
            save_name = f'step-{self.state.global_step}-predictions.jsonl'
            save_path = os.path.join(self.args.output_dir, save_name)
            with jsonlines.open(save_path, 'w') as fwrite:
                fwrite.write_all(predictions)
        return output
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        TRAINING_ARGS_NAME = "training_args.bin"
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # save to sentence transformers
        word_embedding_model = models.Transformer(output_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.args.pooling_type)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
        model.save(output_dir)
