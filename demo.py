from langchain.document_loaders import *
from langchain.text_splitter import *
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import AutoTokenizer

from open_rqa.pipelines.retrieval_qa import SimpleRQA, AutoRQA
from open_rqa.schema.dialogue import DialogueSession
from open_rqa.guardrails.base import NoopAnswerGuardrail
from open_rqa.retrievers.base import BaseRetriever, DummyRetriever
from open_rqa.retrievers.faiss_retriever import FaissRetriever
from open_rqa.text_loaders.langchain_text_loader import LangChainTextLoader, DirectoryTextLoader
from open_rqa.qa_llms.huggingface import HuggingFaceQAModel
import os



if __name__ == "__main__":
    ###### Manual usage of RQA ######
    # a quick way to load data into retriever
    documents = DirectoryTextLoader("/local2/data/shared/rqa/data").load_data()

    # # TODO: this does not work because save_folder was empty
    # # Customized way by using different document_loaders provided in LangChain 
    # loader_func, split_func = DirectoryLoader, CharacterTextSplitter
    # loader_parameters = {'path': "/local2/data/shared/rqa/data", 'glob': "**/*.txt"}
    # splitter_parameters = {'chunk_size': 500, 'chunk_overlap': 200, 'separator': "\n\n"}
    # kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    # documents = LangChainTextLoader(loader_func, split_func).load_data(**kwargs)

    # retriever: BaseRetriever = DummyRetriever()

    # FaissRetriever by using different embeddings provided in Langchain, the embedding index will be cached for faster later use case
    retriever = FaissRetriever(
        documents,
        embeddings=OpenAIEmbeddings(
            model='text-embedding-ada-002',
            organization=os.environ['OPENAI_ORGANIZATION']
        ),
        index_path="/local2/data/shared/rqa/index"
    )
    
    # testcase
    kwargs = {'batch_query': ['what does Revvo do?'], 'batch_dialogue_history': [DialogueSession()]}
    output = retriever.retrieve(**kwargs)
    print(output.batch_source_documents)

    # pick a QA model
    qa_llm = HuggingFaceQAModel(model_name_or_path="lmsys/vicuna-13b-v1.3")
    # pick an answer guardrail

    answer_guardrail = NoopAnswerGuardrail()

    # QA!
    rqa = SimpleRQA(retriever=retriever, qa_llm=qa_llm, answer_guardrail=answer_guardrail, verbose=True)

    response = rqa.qa(
        batch_questions=['what does Revvo do?'],
        batch_dialogue_session=[DialogueSession()],
    )
    print(response.batch_answers[0])

    # ##### Auto usage of RQA ######
    # documents = SimpleDirectoryReader("data").load_data()
    # rqa = AutoRQA(
    #     documents,
    #     retriever="e5",
    #     qa_llm="llama-2",
    # )

    # response = rqa.qa(
    #     batch_questions=["What is the capital of the United States?"],
    #     batch_dialogue_session=[DialogueSession()],
    # )
    # print(response)
