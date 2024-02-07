from langchain.document_loaders import *
from langchain.text_splitter import *
from langchain.embeddings import *
from local_rqa.pipelines.retrieval_qa import SimpleRQA
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.guardrails.base import NoopAnswerGuardrail
from local_rqa.retrievers.faiss_retriever import FaissRetriever
from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader
from local_rqa.qa_llms.huggingface import HuggingFaceQAModel
from local_rqa.utils import init_logger
import logging
import torch


logger: logging.Logger


def full():
    ###### Manual usage of RQA ######
    # a quick way to load data into retriever
    # documents = DirectoryTextLoader("/local2/data/shared/rqa/data").load_data()

    # Customized way by using different document_loaders provided in LangChain 
    loader_func, splitter_func = DirectoryLoader, CharacterTextSplitter
    loader_parameters = {'path': "/local2/data/shared/rqa/data", 'glob': "**/*.txt"}
    splitter_parameters = {'chunk_size': 500, 'chunk_overlap': 200, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    documents = LangChainTextLoader(loader_func=loader_func, splitter_func=splitter_func).load_data(**kwargs)

    # retriever: BaseRetriever = DummyRetriever()

    # FaissRetriever by using different embeddings provided in Langchain, the embedding index will be cached for faster later use case
    # retriever = FaissRetriever(
    #     documents,
    #     embeddings=OpenAIEmbeddings(
    #         model='text-embedding-ada-002',
    #         organization=os.environ['OPENAI_ORGANIZATION']
    #     ),
    #     index_path="/local2/data/shared/rqa/index"
    # )

    # Finetuned
    retriever = FaissRetriever(
        documents,
        embeddings=HuggingFaceEmbeddings(
            model_name="/local2/data/shared/rqa/retriever_model/contriever-ms_databricks_inbatch256_hard0.05_test/checkpoint-300"
        ),
        index_path="/local2/data/shared/rqa/index"
    )

    
    # testcase
    output = retriever.retrieve(batch_questions=['What are some changes in behavior in Databricks Runtime 8.0?'])
    print(output.batch_source_documents)

    # pick a QA model
    qa_llm = HuggingFaceQAModel(model_name_or_path="lmsys/vicuna-13b-v1.5-16k")
    # pick an answer guardrail

    answer_guardrail = NoopAnswerGuardrail()

    # QA!
    rqa = SimpleRQA(retriever=retriever, qa_llm=qa_llm, answer_guardrail=answer_guardrail, verbose=True)

    response = rqa.qa(
        batch_questions=['what does Revvo do?'],
        batch_dialogue_session=[DialogueSession()],
    )
    print(response.batch_answers[0])
    return


def quickstart():
    qa_model_init_args = {
        'torch_dtype': torch.float16,
    }
    rqa = SimpleRQA.from_scratch(
        document_path="data/database/databricks/databricks_400.pkl",
        index_path="data/database/databricks/databricks_400_e5-base-v2",
        embedding_model_name_or_path="intfloat/e5-base-v2",
        qa_model_name_or_path="lmsys/vicuna-7b-v1.5",
        qa_model_init_kwargs=qa_model_init_args,
        qa_is_fid=False,
        verbose=True,
    )
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    print(response.batch_answers[0])
    return


def quickstart_vllm():
    rqa = SimpleRQA.from_scratch(
        document_path="data/database/databricks/databricks_400.pkl",
        index_path="data/database/databricks/databricks_400_e5-base-v2",
        embedding_model_name_or_path="intfloat/e5-base-v2",
        qa_model_name_or_path="vllm::http://localhost:8000/generate",  # where 'http://localhost:8000/generate' is the address of the vllm server
        qa_model_init_kwargs={},
        qa_is_fid=False,
        verbose=True,
    )
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    print(response.batch_answers[0])
    return


def auto():
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
    return

if __name__ == "__main__":
    logger = init_logger()

    # full()
    quickstart()
    auto()