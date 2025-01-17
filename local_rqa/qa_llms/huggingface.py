from typing import List, Optional
from local_rqa.qa_llms.base import BaseQAModel, GenerationOutput
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.schema.document import Document
from local_rqa.qa_llms.prompts import RQA_PROMPT
from local_rqa.qa_llms.fid import FiDT5
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class HuggingFaceQAModel(BaseQAModel):
    """retrieval qa model using huggingface transformers by simply prompting

    Args:
        BaseQAModel (_type_): _description_
    """
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name_or_path: str = "",  # either model or model_name_or_path must be provided
        model_init_kwargs: Optional[dict] = None,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        sep_user = " ",
        sep_sys = "</s>",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        model_init_kwargs = {} if model_init_kwargs is None else model_init_kwargs
        self.model, self.tokenizer = self._init(model, tokenizer, model_name_or_path, model_init_kwargs)

        ## configs
        self._default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "num_beams": 1,
            "eos_token_id": self.tokenizer.eos_token_id, 
        }
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.sep_user = sep_user
        self.sep_sys = sep_sys
        return

    def _init(self, model, tokenizer, model_name_or_path: str, model_init_kwargs: dict):
        if tokenizer is None and model_name_or_path == "":
            raise ValueError("Either tokenizer or model_name_or_path must be provided")
        if model is None and model_name_or_path == "":
            raise ValueError("Either model or model_name_or_path must be provided")
        
        ### intialize model
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **model_init_kwargs)
            model.eval()
        if not next(model.parameters()).is_cuda:
            model = model.cuda()

        ### initialize tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # left for generation
        return model, tokenizer

    def generate(self, batched_prompts: List[str], tokenization_kwargs: dict, generation_kwargs: dict) -> GenerationOutput:
        # merge default kwargs with user kwargs
        _gen_kwargs = {
            **self._default_generate_kwargs,
            **generation_kwargs
        }
        _token_kwargs = {
            "padding": "longest",  # default
            "return_tensors": "pt",
            **tokenization_kwargs
        }

        # generate
        encoded_qs = self.tokenizer(
            batched_prompts, **_token_kwargs
        )
        if 'token_type_ids' in encoded_qs:
            encoded_qs.pop('token_type_ids', None)
        for k, v in encoded_qs.items():
            encoded_qs[k] = v.to(self.model.device)

        output = self.model.generate(
            **encoded_qs,
            **_gen_kwargs
        )

        input_length = encoded_qs['input_ids'].shape[1]
        generated_output = output[:, input_length:].cpu()
        decoded_output = self.tokenizer.batch_decode(
            generated_output,
            skip_special_tokens=True
        )
        return GenerationOutput(
            batch_answers=decoded_output,
        )

    def _prepare_question_w_docs(self, question: str, docs: List[Document], chat_history_str: str):
        # format documents
        formatted_documents = ""
        for doc in docs:
            formatted_documents += f"{doc.fmt_content}\n"
        formatted_documents = formatted_documents.strip()

        formatted_chat = f"{chat_history_str}{self.user_prefix}: {question}{self.sep_user}".strip()
        # format source augmented question
        prompt = RQA_PROMPT.format(
            formatted_documents = formatted_documents,
            formatted_chat = formatted_chat,
            assistant_prefix = self.assistant_prefix,
        )
        return prompt

    def r_generate(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_session: List[DialogueSession],
        tokenization_kwargs: Optional[dict] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> GenerationOutput:
        if tokenization_kwargs is None:
            tokenization_kwargs = {}
        if generation_kwargs is None:
            generation_kwargs = {}
        
        q_w_retrieved_docs = []
        chat_history_strs = [session.to_string() for session in batch_dialogue_session]
        for i, _ in enumerate(batch_questions):
            question = batch_questions[i]
            chat_history_str = chat_history_strs[i]
            docs = batch_source_documents[i]
            augmented_q = self._prepare_question_w_docs(
                question=question,
                docs=docs,
                chat_history_str=chat_history_str
            )
            q_w_retrieved_docs.append(augmented_q)

        answers = self.generate(
            batched_prompts=q_w_retrieved_docs,
            tokenization_kwargs=tokenization_kwargs,
            generation_kwargs=generation_kwargs
        )
        return answers


class HuggingFaceFiDQAModel(BaseQAModel):
    """retrieval qa using fusion in decoder. For now we only support architectures based on T5
    """
    def __init__(self,
        model: Optional[FiDT5] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name_or_path: str = "",  # either model or model_name_or_path must be provided
        model_init_kwargs: Optional[dict] = None,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        sep_user = " ",
        sep_sys = "</s>",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        model_init_kwargs = {} if model_init_kwargs is None else model_init_kwargs
        self.model, self.tokenizer = self._init(model, tokenizer, model_name_or_path, model_init_kwargs)

        ## configs
        self._default_generate_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,  # FiD does not support sampling
        }
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.sep_user = sep_user
        self.sep_sys = sep_sys
        self.fid_token = "<fid>"
        return
    
    def _init(self, model, tokenizer, model_name_or_path: str, model_init_kwargs: dict):
        if tokenizer is None and model_name_or_path == "":
            raise ValueError("Either tokenizer or model_name_or_path must be provided")
        if model is None and model_name_or_path == "":
            raise ValueError("Either model or model_name_or_path must be provided")
        
        ### intialize model
        if model is None:
            model, loading_info = FiDT5.from_pretrained(
                model_name_or_path,
                output_loading_info=True,
                **model_init_kwargs
            )
            if len(loading_info['missing_keys']) > 0:
                del model
                print("Found missing keys, loading with FiDT5.from_t5")
                model = FiDT5.from_t5(model_name_or_path, **model_init_kwargs)
            model.eval()
        if not next(model.parameters()).is_cuda:
            model = model.cuda()

        ### initialize tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # left for generation
        return model, tokenizer

    def pack_fid_inputs(self, batched_fid_prompts: List[List[str]]) -> List[str]:
        packed_fid_inputs = []
        for fid_prompts in batched_fid_prompts:
            packed_fid_inputs.append(self.fid_token.join(fid_prompts))
        return packed_fid_inputs

    def unpack_fid_inputs(self, batched_prompts: List[str]) -> List[List[str]]:
        unpacked_fid_inputs = []
        for prompt in batched_prompts:
            splitted_prompt = prompt.split(self.fid_token)
            unpacked_fid_inputs.append(splitted_prompt)
        # check if length of unpacked prompts are consistent
        assert len(unpacked_fid_inputs[0]) == len(unpacked_fid_inputs[-1])
        return unpacked_fid_inputs

    def encode_fid_inputs(self, batch_q_w_passages, max_length):
        passage_ids, passage_masks = [], []
        for q_w_passsages in batch_q_w_passages:
            # 2D input
            p = self.tokenizer.batch_encode_plus(
                q_w_passsages,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            # 3D input, so that during training we have (batch_size, num_passages, max_length)
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids, passage_masks.bool()
    
    def generate(self, batched_prompts: List[str], tokenization_kwargs: dict, generation_kwargs: dict) -> GenerationOutput:
        # For FiD, we assume ONE str is a concatenated version of FiD input, or just a single question
        unpacked_fid_inputs = self.unpack_fid_inputs(batched_prompts)

        # merge default kwargs with user kwargs
        _gen_kwargs = {
            **self._default_generate_kwargs,
            **generation_kwargs
        }
        _gen_kwargs['do_sample'] = False  # force to False
        
        _token_kwargs = {
            "padding": "longest",  # default
            "return_tensors": "pt",
            "max_length": 512,
            **tokenization_kwargs
        }

        # generate
        input_ids, attention_mask = self.encode_fid_inputs(unpacked_fid_inputs, max_length=_token_kwargs['max_length'])
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **_gen_kwargs
        )
        decoded_output = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True
        )
        return GenerationOutput(
            batch_answers=decoded_output,
        )

    def _prepare_question_w_docs(self, question: str, docs: List[Document], chat_history_str: str) -> List[str]:
        # format documents
        # FiD = append question to each document
        fid_input = []
        formatted_chat = f"{chat_history_str}{self.user_prefix}: {question}{self.sep_user}".strip()
        for doc in docs:
            prompt = RQA_PROMPT.format(
                formatted_documents = doc.fmt_content,
                formatted_chat = formatted_chat,
                assistant_prefix = self.assistant_prefix,
            )
            fid_input.append(prompt)
        return fid_input

    def r_generate(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_session: List[DialogueSession],
        tokenization_kwargs: Optional[dict] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> GenerationOutput:
        if tokenization_kwargs is None:
            tokenization_kwargs = {}
        if generation_kwargs is None:
            generation_kwargs = {}
        
        q_w_retrieved_docs: List[List[str]] = []
        chat_history_strs = [session.to_string() for session in batch_dialogue_session]
        for i, _ in enumerate(batch_questions):
            question = batch_questions[i]
            chat_history_str = chat_history_strs[i]
            docs = batch_source_documents[i]
            fid_augmented_q = self._prepare_question_w_docs(
                question=question,
                docs=docs,
                chat_history_str=chat_history_str
            )
            q_w_retrieved_docs.append(fid_augmented_q)
        
        # pack this
        packed_q_w_retrieved_docs = self.pack_fid_inputs(q_w_retrieved_docs)

        answers = self.generate(
            batched_prompts=packed_q_w_retrieved_docs,
            tokenization_kwargs=tokenization_kwargs,
            generation_kwargs=generation_kwargs
        )
        return answers