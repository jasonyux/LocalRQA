.. _quickstart:

Quickstart
==========

In this quickstart we'll show you:

- How to prepare a document database for your RQA system
- How to use ``SimpleRQA`` to quickly configure and run an RQA system

As a reference, the full example code can be found in ``demo.py`` script at the root of the repository.


Prepare Document
----------------

LocalRQA integrates with frameworks such as LangChain and LlamaIndex to easily ingest text data in various formats, such as JSON data, HTML data, data from Google Drive, etc.

For example, you could load data from a website using ``SeleniumURLLoader`` from ``langchain``, then save and parse them into a collection of documents (``docs``):

.. code-block:: python

    from langchain_community.document_loaders import SeleniumURLLoader
    from langchain.text_splitter import CharacterTextSplitter
    from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader

    # specify how to load the data and how to chunk them
    loader_func, split_func = SeleniumURLLoader, CharacterTextSplitter
    loader_parameters = {'urls': ["https://docs.databricks.com/en/dbfs/index.html"]}
    splitter_parameters = {'chunk_size': 400, 'chunk_overlap': 50, 'separator': "\n\n"}
    kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}

    # load the data, chunk them, and save them
    docs = LangChainTextLoader(
        save_folder="<example>",  # change this to your own folder
        save_filename="<documents.pkl>",
        loader_func=loader_func,
        splitter_func=split_func
    ).load_data(**kwargs)


this list of documents (``docs``) is now your **document database**, which will be used to create an embedding index for the RQA system.


Build an RQA System
----------------
Given a path to a document database (see above), we can directly use ``SimpleRQA`` to 1) create and save an embedding index if ``<example/index>`` is empty, 2) plugin an embedding model and a generative model, and 3) run QA!

.. code-block:: python

    from local_rqa.pipelines.retrieval_qa import SimpleRQA
    from local_rqa.schema.dialogue import DialogueSession

    rqa = SimpleRQA.from_scratch(
        document_path="<example/documents.pkl>",
        index_path="<example/index>",
        embedding_model_name_or_path="intfloat/e5-base-v2",  # embedding model
        qa_model_name_or_path="lmsys/vicuna-7b-v1.5"  # generative model
    )
    response = rqa.qa(
        batch_questions=['What is DBFS?'],
        batch_dialogue_session=[DialogueSession()],
    )
    print(response.batch_answers[0])
    # DBFS stands for Databricks File System, which is a ...


where ``response`` is a ``RQAOutput`` object:

.. code-block:: python

    class RQAOutput:
        batch_answers: List[str]
        batch_source_documents: List[List[Document]]
        batch_dialogue_session: List[DialogueSession]


Next Steps
--------

Beyond this simple example, you can:

- prepare your own RQA data for training, evaluation, and serving (:ref:`data-main`)
- train your own models using algorithms we implemented from latest research (:ref:`train-main`)
- evaluate your RQA system with automatic metrics (:ref:`evaluation-main`)
- deploy your RQA system to interact with real users (:ref:`serving-main`)