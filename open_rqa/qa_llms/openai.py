from openai import OpenAI
from typing import List, Optional
from open_rqa.qa_llms.base import BaseQAModel, GenerationOutput
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.schema.document import Document
from open_rqa.qa_llms.prompts import RQA_PROMPT
from open_rqa.constants import QA_ERROR_MSG
import os


class OpenAIQAModel(BaseQAModel):
    def __init__(
        self,
        model_name: str,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        sep_user: str = " ",
        sep_sys: str = " "
    ) -> None:
        super().__init__()
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY"),
            organization = os.environ.get("OPENAI_ORGANIZATION")
        )
        self.model_name = model_name
        self._default_generate_kwargs = {
            "temperature": 0.7,
            "timeout": 30.0,
        }
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.sep_user = sep_user
        self.sep_sys = sep_sys
        return

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

    def generate(self, batched_prompts: List[str], tokenization_kwargs: dict, generation_kwargs: dict) -> GenerationOutput:
        responses = []
        _gen_kwargs = {
            **self._default_generate_kwargs,
            **generation_kwargs
        }
        for prompt in batched_prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    **_gen_kwargs
                )
                extracted_message = response.choices[0].message.content
            except Exception as e:
                print(e)
                extracted_message = QA_ERROR_MSG
            responses.append(extracted_message)
        return GenerationOutput(
            batch_answers=responses,
        )