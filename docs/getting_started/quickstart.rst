quickstart
==========

In this quickstart we'll show you how to:

- How to prepare documents for retrieval
- How to use ``SimpleRQA`` to get the response based on the existing embedding model and generative model

You could also run ``quickstart()`` function in ``demo.py`` script to directly start the pipeline.


Prepare Document
----------------

LocalRQA supports integration with frameworks such as LangChain and LlamaIndex to easily ingest text data in various formats, such as JSON data, HTML data, data from Google Drive, etc.

For example, you could prepare document from website url
::
    from langchain_community.document_loaders import SeleniumURLLoader
    from langchain.text_splitter import CharacterTextSplitter
    from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader

    loader_func, split_func = SeleniumURLLoader, CharacterTextSplitter
    loader_parameters = {'urls': ["<website_url>"]}
    splitter_parameters = {'chunk_size': 400, 'chunk_overlap': 50, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
    docs = LangChainTextLoader(save_folder="<path/to/save/dir>", save_filename="<filename>", loader_func=loader_func, splitter_func=split_func).load_data(**kwargs)


Pipeline
--------
By providing document that generated in the previous step, and specifying the ``embedding_model_name_or_path`` and ``qa_model_name_or_path``, the LocalRQA pipeline is ready to go!
::
    rqa = SimpleRQA.from_scratch(
        document_path="path/to/save/dir/filename",
        index_path="path/to/index",
        embedding_model_name_or_path="intfloat/e5-base-v2",
        qa_model_name_or_path="lmsys/vicuna-7b-v1.5"
    )
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    final_answer = response.batch_answers[0]

where ``response`` is a ``RQAOutput`` object with the following construction::

    class RQAOutput:
        """stores the answers to a user's question, the relevant source documents, and the UPDATED dialogue history"""
        batch_answers: List[str]
        batch_source_documents: List[List[Document]]
        batch_dialogue_session: List[DialogueSession]

