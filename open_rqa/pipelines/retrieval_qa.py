from typing import Any, Dict, List
from open_rqa.schema.document import Document
from open_rqa.schema.dialogue import DialogueSession, RQAOutput
from open_rqa.guardrails.base import BaseAnswerGuardrail
from open_rqa.retrievers.base import BaseRetriever
from open_rqa.qa_llms.base import BaseQAModel
from open_rqa.pipelines.base import RQAPipeline
import torch
import logging


logger = logging.getLogger(__name__)


class BaseRQA(RQAPipeline):
    def __init__(self,
        retriever: BaseRetriever,
        qa_llm: BaseQAModel,
        answer_guardrail: BaseAnswerGuardrail
    ):
        self.retriever = retriever
        self.qa_llm = qa_llm
        self.guardrail = answer_guardrail
        return

    def qa(self, batch_questions: List[str], batch_dialogue_history: List[DialogueSession]) -> RQAOutput:
        # retrieve relevant documents
        retrieval_output = self.retriever.retrieve(batch_questions, batch_dialogue_history)
        # generate answers
        raw_gen_output = self.qa_llm.r_generate(retrieval_output)
        # guardrail
        gen_output = self.guardrail.guardrail(
            batch_questions=batch_questions,
            batch_source_documents=retrieval_output.source_documents,
            batch_dialogue_history=batch_dialogue_history,
            batch_generated_answers=raw_gen_output.batched_answers
        )
        return gen_output


class AutoRQA(BaseRQA):
    def __init__(self,
        retriever: BaseRetriever,
        qa_llm: BaseQAModel,
        answer_guardrail: BaseAnswerGuardrail
    ):
        super().__init__(retriever, qa_llm, answer_guardrail)
        return
        

class SimpleRQA:
    def __init__(self,
        generation_model: str,
        texts_db_path: str,
        verbose=False,
        **kwargs
    ):
        super().__init__(texts_db_path)
        # init generation model
        tokenizer, model = self._init_model(generation_model)
        self.tokenizer = tokenizer
        self.model = model
        self.is_api_model = tokenizer is None

        # init retriever model
        self._memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)
        retriever_kwargs = kwargs.get('retriever_kwargs', {})
        self._retriever, self.embeddings = self._init_retriever(**retriever_kwargs)

        self.verbose = verbose

        self.output_key = "answer"
        self.user_prefix = "USER"
        self.assistant_prefix = "ASSISTANT"
        self.__eos_token = "</s>"  # used in version 1.1

        # TODO: with TGI we have quite a different score than not using it. perhaps check if it was doing quantization automatically
        self.__default_generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
            # "repetition_penalty": 1.00,  # cause CUDA-assertion when used with TGI
            # "typical_p": 0.999,  # cause CUDA-assertion when used with TGI
            "eos_token_id": None if self.is_api_model else self.tokenizer.eos_token_id, 
            "early_stopping": True,
        }
        return

    @property
    def memory(self) -> ConversationBufferMemory:
        return self._memory
    
    @memory.setter
    def memory(self, value: ConversationBufferMemory):
        self._memory = value
        return
    
    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    def _init_model(self, generation_model):
        if "://" in generation_model:  # hosted by text-generation-inference
            model = TGIAPIModel(generation_model)
            tokenizer = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(generation_model)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            # doesn't work
            # model = AutoModelForCausalLM.from_pretrained(generation_model, load_in_8bit=True)  # 4min generation time
            model = AutoModelForCausalLM.from_pretrained(generation_model)
            model.cuda()
            model.eval()
        return tokenizer, model

    def _init_retriever(self, **kwargs):
        texts = self._load_text_from_db()

        model_name = kwargs.get('model_name', 'facebook/contriever-msmarco')
        model_type = kwargs.get('model_type', 'bert')  # if BERT, emebdding calculated from hidden states. if bert_mlm, embedding from CLS token
        index_path = kwargs.get('index_path', None)
        normalize_L2 = kwargs['search_kwargs'].get('normalize_L2', False)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == 'bert':
            model = BertModel.from_pretrained(model_name)
            embeddings = LocalEmbeddings(model, tokenizer, index_path=index_path, device = "cuda:1")
        elif model_type == 'bert_mlm':
            model = BertForMaskedLM.from_pretrained(model_name)
            embeddings = LocalBERTMLMEmbeddings(model, tokenizer, index_path=index_path, device = "cuda:1")
        else:
            raise NotImplementedError(f"Unknown model type {model_type} for retriever")

        docsearch = MyFAISS.from_documents(texts, embeddings, normalize_L2=normalize_L2)
        
        retriever = docsearch.as_retriever(**kwargs)
        return retriever, embeddings
    
    def _batch_generate(self, input_texts, **generate_kwargs):
        # merge default kwargs with user kwargs
        _gen_kwargs = {
            **self.__default_generate_kwargs,
            **generate_kwargs
        }

        if self.is_api_model:
            cleaned_answers = []
            for input_text in input_texts:
                answer = self.model.generate(input_text, **_gen_kwargs)
                cleaned_answers.append(answer)
        else:
            encoded_qs = self.tokenizer(input_texts, padding='longest', return_tensors="pt")
            encoded_qs.pop('token_type_ids', None)
            for k, v in encoded_qs.items():
                encoded_qs[k] = v.to(self.model.device)

            output = self.model.generate(
                **encoded_qs,
                **_gen_kwargs
            )
            answers = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            cleaned_answers = []
            for answer, input_text in zip(answers, input_texts):
                # remove the input text from the answer
                last_100_input_text = input_text[-100:]
                new_start_idx = answer.find(last_100_input_text) + len(last_100_input_text)
                answer = answer[new_start_idx:].strip()
                answer = answer.replace(self.__eos_token, "")
                cleaned_answers.append(answer)
        return cleaned_answers
    
    def _batch_rephrase_question(self, questions: List[str], chat_history_strs: List[str]):
        prompt = """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        Chat History: {chat_history_str}
        Follow Up Input: {question}{eos_token}
        Standalone question:
        """.replace("\t", "").strip()
        input_prompts = []
        for question, chat_history_str in zip(questions, chat_history_strs):
            prompt_i = prompt.format(
                question=question, 
                chat_history_str=chat_history_str, 
                eos_token=self.__eos_token
            )
            input_prompts.append(prompt_i)
            if self.verbose:
                logger.info("[_rephrase_question] Prompt:")
                logger.info(prompt_i)

        rephrased_qs = self._batch_generate(input_prompts, max_new_tokens=128)
        if self.verbose:
            logger.info("Rephrased questions:")
            logger.info('\n'.join(rephrased_qs))
        return rephrased_qs
    
    def _format_chat_history(self):
        session_history_list = self.memory.buffer
        formatted_chat_history = ""
        for msg in session_history_list:
            if msg.type == 'human':
                formatted_chat_history += f"{self.user_prefix}: {msg.content} "
            elif msg.type == 'ai':
                formatted_chat_history += f"{self.assistant_prefix}: {msg.content}{self.__eos_token}"
            else:
                raise ValueError(f"Unknown message type {msg.type}")
        return formatted_chat_history.strip()
    
    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        docs = self.retriever.get_relevant_documents(question)
        return docs
    
    def _batch_get_docs(self, questions: List[str], inputs: Dict[str, Any]) -> List[List[Document]]:
        all_docs = []
        for question in questions:
            docs = self._get_docs(question, inputs)
            all_docs.append(docs)
        return all_docs
    
    def _prepare_question_w_docs(self, question: str, docs: List[Document], chat_history_str: str):
        # format documents
        formatted_documents = ""
        for doc in docs:
            formatted_documents += f"title: {doc.metadata['title']} content: {doc.page_content}\n"
        formatted_documents = formatted_documents.strip()

        formatted_chat = f"{chat_history_str} {self.user_prefix}: {question}".strip()
        # format source augmented question
        prompt = REPLY_PROMPT.format(
            formatted_documents = formatted_documents,
            formatted_chat = formatted_chat,
            assistant_prefix = self.assistant_prefix,
        )

        if self.verbose:
            logger.info("[_prepare_question_w_docs] Prompt:")
            logger.info(prompt)
        return prompt
    
    def build_index(self, documents: List[Document] = [], format_str='title: {title} content: {text}') -> torch.Tensor:
        passages = []
        for p in documents:
            passages.append(format_str.format(title=p.metadata['title'], text=p.page_content))
        
        if len(passages) == 0:
            return torch.tensor([])
        list_embeddings = self.embeddings.build_index_from_texts(passages)
        # [len(passages), 768]
        embeddings = torch.tensor(list_embeddings)
        return embeddings
    
    def retrieve(self, batch_inputs, passages=None, indices=None) -> RetrievalOutput:
        questions = batch_inputs["question"]
        chat_history_strs = batch_inputs["chat_history_str"]
        rephrased_questions = []

        # batch rephrase questions
        to_rephrase_qids = []
        to_rephrase_questions = []
        to_rephrase_chat_history_strs = []
        for i, question in enumerate(questions):
            chat_history_str_i = chat_history_strs[i]
            if chat_history_str_i == "":
                rephrased_questions.append(question)
            else:
                rephrased_questions.append(None)  # placeholder
                to_rephrase_qids.append(i)
                to_rephrase_questions.append(question)
                to_rephrase_chat_history_strs.append(chat_history_str_i)
        if len(to_rephrase_questions) > 0:
            rephrased_questions = self._batch_rephrase_question(to_rephrase_questions, to_rephrase_chat_history_strs)
            for i, qid in enumerate(to_rephrase_qids):
                rephrased_questions[qid] = rephrased_questions[i]

        # retrieve
        docs = self._batch_get_docs(rephrased_questions, {})
        return RetrievalOutput(
            retrieved_docs=docs,
            other_outputs={
                'chat_history_strs': chat_history_strs,
            }
        )
    
    def generate_from_docs(self, batch_inputs, retr_output: RetrievalOutput) -> GenerationOutput:
        questions_w_all_info = []
        chat_history_strs = retr_output.other_outputs['chat_history_strs']
        for i in range(len(batch_inputs["question"])):
            question = batch_inputs["question"][i]
            chat_history_str = chat_history_strs[i]
            docs = retr_output.retrieved_docs[i]
            question_w_all_info = self._prepare_question_w_docs(question, docs, chat_history_str)
            questions_w_all_info.append(question_w_all_info)

        answers = self._batch_generate(questions_w_all_info)

        return GenerationOutput(
            generated_answers=answers,
            source_documents=retr_output.retrieved_docs,
        )

    def qa(self, inputs: dict):
        if '__len__' not in inputs:
            # single batch
            inputs = {k: [v] for k, v in inputs.items()}
            inputs['__len__'] = 1
        
        retr_output = self.retrieve(inputs)
        gen_output = self.generate_from_docs(inputs, retr_output)

        question = inputs['question'][0]
        answer = gen_output.generated_answers[0]
        docs = gen_output.source_documents[0]
        # memory management is technically done externally since we will only have one model dealing with all the requests
        # essentially, we see the model input as simply (input_question, chat_history
        # this is for backward compatibility
        self._memory.chat_memory.add_user_message(question)
        self._memory.chat_memory.add_ai_message(answer)

        return {self.output_key: answer, "source_documents": docs}

    def __call__(self, inputs: dict) -> Dict[str, Any]:
        text = inputs["text"].strip()
        chat_history = inputs["chat_history"]
        self._memory = self._load_memory(chat_history)
        chat_history_str = self._format_chat_history()

        result = self.qa({
            "question": [text], 
            "chat_history_str": [chat_history_str],
            "__len__": 1
        })

        fmt_result = GenerationOutput(
            generated_answers=[result['answer']],
            source_documents=[result['source_documents']],
        )

        cleaned_result = self.answer_guardrail(fmt_result)
        return_result = self._prepare_output(cleaned_result)


        # remove sources for dontknow answers
        
        # if result['answer'].startswith('Assistant:'):
        # 	result['answer'] = result['answer'][len('Assistant:'):].strip()
        
        # result = self._process_dontknow(result)
        # logger.info('final answer: ' + result['answer'])

        # # remove sources if unused in the response
        # result['source_documents'] = self._check_sources(result['answer'], result['source_documents'])

        # result = self._prepare_output(result)
        return return_result