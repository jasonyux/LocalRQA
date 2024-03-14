.. _data-description:

RQA Data Format
===============

In this section, we provide detailed examples of data files used throughout LocalRQA. We will provide examples of what each of the following files looks like, which appears throughout this documentation:

* training, evaluation, and test files: ``<example/train_w_qa.jsonl>``, ``<example/eval_w_qa.jsonl>``, and ``<example/test_w_qa.jsonl>``
* document database: ``<example/documents.pkl>``
* document index: ``<example/index>``


Training, Evaluation, and Test Files
------------------------------------

Training, evaluation, and test files such as ``<example/train_w_qa.jsonl>``, ``<example/eval_w_qa.jsonl>``, and ``<example/test_w_qa.jsonl>`` are used extensively for :ref:`train-main` and :ref:`evaluation-main`. These files are in JSONL format, where each line contains


Since these three files are in the same format, we provide an example of the training file, ``<example/train_w_qa.jsonl>``:


.. code-block:: jsonl

    {'chat_history': [], 'question': 'xxx?', 'gold_docs': [...], 'gold_answer': 'xxx', 'hard_neg_docs': [...], }
    {'chat_history': [], 'question': 'xxx?', 'gold_docs': [...], 'gold_answer': 'xxx', 'hard_neg_docs': [...], }
    {'chat_history': [], 'question': 'xxx?', 'gold_docs': [...], 'gold_answer': 'xxx', 'hard_neg_docs': [...], }
    ...

where each line represents a single training/evaluation/test example. The fields are as follows:

.. code-block:: json
    
    {
        "chat_history": [],             // (optional) chat history before "question".
        "question": "What is DBFS?",    // a user's question
        "gold_docs": [                  // list of reference documents used to answer the user's question
            {"page_content": "## DBFS\n DBFS is ...", "fmt_content": "...", "metadata": {...}}
        ],
        "gold_answer": "DBFS is ...",   // reference answer to the user's question
        "hard_neg_docs": [              // (optional) list of documents that seem related but do not contain the correct answer
            {"page_content": "## DFS\n Depth First Search ...", "fmt_content": "...", "metadata": {...}},
            {"page_content": "## Normal FS\n File system ...", "fmt_content": "...", "metadata": {...}}
        ]
    }

where:

*  ``chat_history`` represents previous turns in the dialogue, preceding the ``question``. This list can be obtained by ``local_rqa.schema.dialogue.DialogueSession.to_list()`` function. If the dialogue history is not available (e.g., first turn in the dialogue), simply use an empty list.
*  ``gold_docs`` is a list of reference documents that contains the correct answer to the user's question. Most of the time, this list contains only one gold document. To obtain such a document, you can use ``local_rqa.schema.document.Document.to_dict()`` function.
*  ``hard_neg_docs`` is a list of hard negative documents that seem related to the user's question but do not contain the correct answer. This is used by :ref:`training-ret-ctl`, and is hence optional for other cases (simply leave as an empty list).


In practice, you can either manually prepare them and reformat them as above, or refer to :ref:`data-preparation` for methods to automatically prepare these files.


Document Database and Vector Index
---------------------------------

Document database ``<example/documents.pkl>`` and its vector index ``<example/index>`` are used throughout all modules in LocalRQA. They are core components for performing retrieval. In essense, a **document database** is simply a list of documents in the format of ``local_rqa.schema.document.Document``, and a **vector index** simply uses a neural encoder to embed all these documents into vectors.

For example, inside ``<example/documents.pkl>``, you can have:

.. code-block:: python

    [
        Document(page_content="## DBFS\n DBFS is ...", fmt_content="...", metadata={"title": "...", "url": "..."}),
        Document(page_content="## DFS\n Depth First Search ...", fmt_content="...", metadata={"title": "...", "url": "..."}),
        ...
    ]


And then the vector index ``<example/index>`` is a directory that contains the vector version of the ``fmt_content`` field of each document:

.. code-block:: bash

    example/index
    ├── customized000d4bd3-ece0-5cf7-ad01-9e14e1d1e073
    ├── customized0010a3a4-429d-5115-90f3-d01d91a75ea1
    ├── customized00148cdd-6bc8-5c81-aabf-9db00e00e769
    ├── ....
    └── customizedffff1e54-52af-5d1e-a1be-235cc926f21e


Since the vector index essentially uses a neural encoder to process your document database, LocalRQA will **automatically** create such a vector index whenever you see a place asking for ``--index_path`` or ``--full_dataset_index_path``.


.. note::

    The purpose of saving the vector index is to **avoid re-embedding the entire document database** every time you restart the training/testing/serving process. Therefore, if the specified index path is not empty, we will **skip the document embedding step and directly load the index**.
    
    This means that if you really want to re-embed the document database, you should delete the index directory or specify a new index path that is empty.