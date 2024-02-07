from typing import List, Optional
from copy import deepcopy
from retry import retry
from local_rqa.qa_llms.base import BaseQAModel, GenerationOutput
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.schema.document import Document
from local_rqa.qa_llms.prompts import RQA_PROMPT
from local_rqa.constants import QA_ERROR_MSG
from text_generation import Client
import logging


logger = logging.getLogger(__name__)


class TGIQAModel(BaseQAModel):
    """under the hood its using your generative model hosted on Text-Generation-Inference

    Args:
        BaseQAModel (_type_): _description_
    """
    is_api_model = True
    
    def __init__(
        self,
        url,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        sep_user: str = " ",
        sep_sys: str = "</s>"
    ) -> None:
        self.client = Client(url, timeout=60)
        self.url = url
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.sep_user = sep_user
        self.sep_sys = sep_sys

        self._default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "eos_token_id": None,
        }
        return
    
    def prepare_gen_kwargs(self, input_kwargs):
        new_input_kwargs = deepcopy(input_kwargs)
        if 'num_beams' in new_input_kwargs:
            new_input_kwargs.pop('num_beams')
            logger.warning("num_beams is not supported in text_generation_inference")
        
        if 'eos_token_id' in new_input_kwargs:
            new_input_kwargs.pop('eos_token_id')
            new_input_kwargs['stop_sequences'] = ['</s>']
            logger.warning("eos_token_id is not supported in text_generation_inference. Using stop_sequences='</s>' instead.")
        
        if 'early_stopping' in new_input_kwargs:
            new_input_kwargs.pop('early_stopping')
            logger.warning("early_stopping is not supported in text_generation_inference")
        
        return new_input_kwargs

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

    @retry(Exception, tries=3, delay=0.5)
    def _generate(self, input_text, **generate_kwargs):
        generate_kwargs = self.prepare_gen_kwargs(generate_kwargs)
        response = self.client.generate(input_text, **generate_kwargs)
        return response.generated_text
    
    def _generate_stream(self, input_text, **generate_kwargs):
        generate_kwargs = self.prepare_gen_kwargs(generate_kwargs)
        for response in self.client.generate_stream(input_text, **generate_kwargs):
            if not response.token.special:
                yield response.token.text

    def generate(self, batched_prompts: List[str], tokenization_kwargs: dict, generation_kwargs: dict) -> GenerationOutput:
        responses = []
        _gen_kwargs = {
            **self._default_generate_kwargs,
            **generation_kwargs
        }
        _gen_kwargs = self.prepare_gen_kwargs(_gen_kwargs)
        for prompt in batched_prompts:
            try:
                generated_text = self._generate(prompt, **_gen_kwargs)
            except Exception as e:
                print(e)
                generated_text = QA_ERROR_MSG
            responses.append(generated_text)
        return GenerationOutput(
            batch_answers=responses,
        )