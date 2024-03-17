import logging
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from local_rqa.pipelines.retrieval_qa import SimpleRQA
from local_rqa.schema.dialogue import DialogueSession
from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader
from local_rqa.utils import init_logger


logger: logging.Logger


def quickstart():
    ## note: requires selenium to read the web page
    ## you can also skip this entire section, as we have provided the `example/demo/databricks_web.pkl` file as well
    loader_func, split_func = SeleniumURLLoader, CharacterTextSplitter
    loader_parameters = {'urls': ["https://docs.databricks.com/en/dbfs/index.html"]}
    splitter_parameters = {'chunk_size': 400, 'chunk_overlap': 50, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    docs = LangChainTextLoader(
        save_folder="example/demo",
        save_filename="databricks_web",
        loader_func=loader_func,
        splitter_func=split_func
    ).load_data(**kwargs)

    ## run QA!
    rqa = SimpleRQA.from_scratch(
        document_path="example/demo/databricks_web.pkl",
        index_path="example/demo/index",
        embedding_model_name_or_path="intfloat/e5-base-v2",
        qa_model_name_or_path="lmsys/vicuna-7b-v1.5"
    )
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    # print(response)
    print(response.batch_answers[0])
    return


def quickstart_vllm():
    ## an example using VLLM with the LocalRQA
    ## this requires you to first host the VLLM server at 'http://localhost:8000/generate'
    rqa = SimpleRQA.from_scratch(
        document_path="example/demo/databricks_web.pkl",
        index_path="example/demo/index",
        embedding_model_name_or_path="intfloat/e5-base-v2",
        qa_model_name_or_path="vllm::http://localhost:8000/generate",  # where 'http://localhost:8000/generate' is the address of the vllm server
    )
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    # print(response)
    print(response.batch_answers[0])
    return


if __name__ == "__main__":
    logger = init_logger()
    quickstart()
