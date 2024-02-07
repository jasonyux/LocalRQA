import json
import requests
import logging
from retry import retry
from local_rqa.qa_llms.base import BaseQAModel, GenerationOutput
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.schema.document import Document
from local_rqa.qa_llms.prompts import RQA_PROMPT
from local_rqa.constants import QA_ERROR_MSG
from typing import Iterable, List, Optional


logger = logging.getLogger(__name__)


class SGLangClient:
    def __init__(self, url, timeout=60) -> None:
        self.url = url
        self.timeout = timeout
        return

    def _post_http_request(
        self,
        prompt: str,
        stream: bool = False,
        **gen_args
    ) -> requests.Response:
        pload = {
            "text": prompt,
            "sampling_params": gen_args,
            "stream": stream,
        }
        response = requests.post(
            self.url,
            json=pload,
            stream=stream,
        )
        return response

    def _get_streaming_response(self, response: requests.Response) -> Iterable[List[str]]:
        prev = 0
        for chunk in response.iter_lines(chunk_size=8192,
                                        decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][prev:]
                yield output
                prev += len(output)

    def _get_response(self, response: requests.Response) -> List[str]:
        data = response.json()
        output = data["text"]
        return output

    def generate(self, input_text: str, **generate_kwargs):
        response = self._post_http_request(input_text, **generate_kwargs)
        generated_text = self._get_response(response)
        return generated_text

    def generate_stream(self, input_text: str, **generate_kwargs):
        response = self._post_http_request(input_text, stream=True, **generate_kwargs)
        for token in self._get_streaming_response(response):
            yield token


class SGLangQAModel(BaseQAModel):
    is_api_model = True
    
    def __init__(
        self,
        url,
        user_prefix: str = "USER",
        assistant_prefix: str = "ASSISTANT",
        sep_user: str = " ",
        sep_sys: str = "</s>"
    ) -> None:
        self.client = SGLangClient(url, timeout=60)
        self.url = url
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.sep_user = sep_user
        self.sep_sys = sep_sys

        self._default_generate_kwargs = {
            "temperature": 0.7,
            "max_new_tokens": 256,
            "stop": ["</s>"]
        }
        self._allowed_params = [
            'max_new_tokens', 'stop', 'temperature', 'top_p', 'top_k', 'frequency_penalty', 'presence_penalty'
        ]
        return

    def prepare_gen_kwargs(self, input_kwargs):
        # print some warning to let user know what is not supported
        if 'num_beams' in input_kwargs:
            input_kwargs.pop('num_beams')
            logger.warning("num_beams is not supported in sglang")
        if 'eos_token_id' in input_kwargs:
            input_kwargs.pop('eos_token_id')
            logger.warning("eos_token_id is not supported in sglang. Using stop instead.")
        if 'early_stopping' in input_kwargs:
            input_kwargs.pop('early_stopping')
            logger.warning("early_stopping is not supported in sglang")
        if 'do_sample' in input_kwargs:
            input_kwargs.pop('do_sample')
            logger.warning("do_sample is not supported in sglang, use temperature instead")
        if 'stream' in input_kwargs:
            input_kwargs.pop('stream')
        new_input_kwargs = {}
        for k in self._allowed_params:
            if k in input_kwargs:
                new_input_kwargs[k] = input_kwargs[k]
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
        generated_text = self.client.generate(input_text, **generate_kwargs)
        return generated_text
    
    def _generate_stream(self, input_text, **generate_kwargs):
        generate_kwargs = self.prepare_gen_kwargs(generate_kwargs)
        for token in self.client.generate_stream(input_text, **generate_kwargs):
            yield token

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


if __name__ == "__main__":
    # example usage
    # assume you already have a running vllm server
    # e.g.: python -m sglang.launch_server --model-path lmsys/vicuna-7b-v1.5 --port 30000 will host at http://localhost:30000/generate
    rqa_model = SGLangQAModel(url="http://localhost:30000/generate")
    question = "Tell me a very long story."
    output = rqa_model.generate(
        batched_prompts=[question],
        tokenization_kwargs={},
        generation_kwargs={'max_new_tokens': 32}
    )
    print('[[not streaming]]')
    print(output.batch_answers[0])

    ## stream
    print('[[streaming]]')
    for token in rqa_model._generate_stream(question, max_new_tokens=32):
        print(token, end="", flush=True)