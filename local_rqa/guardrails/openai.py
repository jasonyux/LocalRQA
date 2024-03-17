from typing import List
from local_rqa.schema.document import Document
from local_rqa.schema.dialogue import DialogueSession, RQAOutput
from local_rqa.guardrails.base import BaseAnswerGuardrail
import os
import requests


class OpenAIModeration(BaseAnswerGuardrail):
    """checks if the answer violates OpenAI moderation API."""
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    def _moderate_single(self, text: str) -> bool:
        url = "https://api.openai.com/v1/moderations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.OPENAI_API_KEY
        }
        text = text.replace("\n", "")
        json_data = {'input': text}
        try:
            ret = requests.post(url, headers=headers, json=json_data, timeout=5)
            flagged = ret.json()["results"][0]["flagged"]
        except requests.exceptions.RequestException as _:
            flagged = False
        except KeyError as _:
            flagged = False
        return flagged

    def guardrail(
        self,
        batch_questions: List[str],
        batch_source_documents: List[List[Document]],
        batch_dialogue_session: List[DialogueSession],
        batch_answers: List[str],
    ) -> RQAOutput:
        checked_answers = []
        checked_source_documents = []
        for idx, answer in enumerate(batch_answers):
            if self._moderate_single(answer):
                checked_answers.append("I'm sorry, I cannot answer that question.")
                checked_source_documents.append([])
            else:
                checked_answers.append(answer)
                checked_source_documents.append(batch_source_documents[idx])

        return RQAOutput(
            batch_answers=checked_answers,
            batch_source_documents=checked_source_documents,
            batch_dialogue_session=batch_dialogue_session,
        )