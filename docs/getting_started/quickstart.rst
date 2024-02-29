quickstart
==========

In this quickstart we'll show you how to:

- How to prepare documents for retrieval
- How to use ``SimpleRQA`` to get the response based on the existing embedding model and generative model


Prepare Document
----------------

LocalRQA supports integration with frameworks such as LangChain and LlamaIndex to easily ingest text data in various formats, such as JSON data, HTML data, data from Google Drive, etc.

For example, you could prepare document from website url
::
    from langchain.document_loaders import *
    from langchain.text_splitter import *
    from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader

    loader_func, split_func = SeleniumURLLoader, CharacterTextSplitter
    loader_parameters = {'urls': ["<website_url>"]}
    splitter_parameters = {'chunk_size': 400, 'chunk_overlap': 50, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    docs = LangChainTextLoader(save_folder="<path/to/save/dir>", save_filename="<filename>", loader_func, split_func).load_data(**kwargs)


Pipeline
--------
By providing document that generated in the previous step, and specifying the ``embedding_model_name_or_path`` and ``qa_model_name_or_path``, the LocalRQA pipeline is ready to go!
::
    rqa = SimpleRQA.from_scratch(
        document_path="path/to/save/dir/filename",
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
    final_answer = response.batch_answers[0]
where ``final_answer`` is the response after post process.