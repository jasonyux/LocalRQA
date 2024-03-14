.. _data-main:

Data
====

* Explain on a high level what kind of data we will using for training, evaluation, and serving.
* Explain how we can obtain the data briefly, and point to the relevant subpages.
* Provide a high level example.


Training, evaluating, or serving an RQA system requires data. Depending on your use case, what data you need to prepare **will vary**. In a standard workflow with full model training → evaluation → deployment, you will need: 1) a **collection of documents** acting as a database, and 2) a collection of **question-passage-answer triplets** used for training and evaluation.


.. figure:: /_static/data/sample-flow.png
    :align: center
    :width: 55 %
    :alt: Example workflow using LocalRQA

    Example workflow using LocalRQA. The data components are colored in **blue**.


As an example, a (simplified) document database may look like this:

.. code-block:: json

    [
        {
            "content": "Configure access to the `default.people10m` table\n\nEnable the user you created ...",
            "metadata": {
                "title": "...",
                "url": "..."
            }
        },
        {
            "content": "## Updated required permissions for new workspaces\n\n  **February 28, 2023**\n\n  The [set of required permissions] ...",
            "metadata": {
                "title": "...",
                "url": "..."
            }
        }
        ...
    ]


and (simplified) examples of question-passage-answer triplets may look like this:


.. code-block:: json

    [
        {
            "question": "How can I enable a user to access a specific table in Databricks?",
            "passage": "Configure access to the `default.people10m` table\n\nEnable the user you created ...",
            "answer": "To enable a user to access a specific table in Databricks, you can follow these steps: ..."
        },
        {
            "question": "How can I access ADLS2 with UC external locations in Databricks?"
            "passage": "## Access <ADLS2> with <UC> external locations\n\n  .. note:: <ADLS2> is ...",
            "answer": "You can access Azure Data Lake Storage Gen2 (ADLS2) with Unity Catalog (UC) by ..."
        }
    ]


What Data Do We Need?
----------------------

Depending on how you wish to use LocalRQA, you may **not** need all the data mentioned above. In this section, we provide high-level explanations of the kind of data you will use for training, evaluation, and serving an RQA system.


Training Data
~~~~~~~~~~~~~

Training an RQA system **end-to-end** requires a collection of **question-passage-answer triplets**. Typically:

* to train a retrieval model, you would need a collection of (question, passage) pairs
* to train a generative model, you would need a collection of (question, passage, answer) triplets

However, the exact requirement depends on **what training algorithms you want to use**. For more details, you can refer to :ref:`train-main`.


Evaluation Data
~~~~~~~~~~~~~~~

LocalRQA provides two ways to evaluate an RQA system: automatic evaluation (see :ref:`evaluation-main`) and human evaluation (see :ref:`serving-human-eval`)

When evaluating an RQA system, you will need a collection of **question-passage-answer triplets**. These triplets are used to evaluate the model's performance. The question-passage-answer triplets can be obtained from various sources, such as:


Serving Data
~~~~~~~~~~~~

When serving an RQA system, you will need a collection of **documents** acting as a database.


Data Schema
-----------

How does LocalRQA expect the data to be formatted?


How to Obtain RQA Data?
------------------------

AA


.. toctree::
    :maxdepth: 1
    :caption: Data
    :hidden:

    data/data_description.rst
    data/data_preparation.rst