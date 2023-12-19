from typing import List, Optional
from open_rqa.qa_llms.base import BaseQAModel, GenerationOutput
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.schema.document import Document
from open_rqa.qa_llms.prompts import RQA_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self._init(model, tokenizer, model_name_or_path)
        ## configs
        self._default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": self.tokenizer.eos_token_id, 
            "early_stopping": True,
        }
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        return

    def _init(self, model, tokenizer, model_name_or_path: str):
        if tokenizer is None and model_name_or_path == "":
            raise ValueError("Either tokenizer or model_name_or_path must be provided")
        if model is None and model_name_or_path == "":
            raise ValueError("Either model or model_name_or_path must be provided")
        
        # intialize model
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            model.cuda()
            model.eval()
        # initialize tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
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

        formatted_chat = f"{chat_history_str} {self.user_prefix}: {question}".strip()
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
        return GenerationOutput(
            batch_answers=answers.batch_answers,
        )


class HuggingFaceFiDQAModel(BaseQAModel):
    """retrieval qa using fusion in decoder
    """
    pass