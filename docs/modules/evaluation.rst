.. _evaluation-main:

Evaluation
==========


Given an RQA system, LocalRQA implements many automatic evaluation metrics to help measure its performance. The resulting scores can be used to compare different system configurations, so that you can find the most cost-effective models/training methods suitable for your applications.

We provide standalone scripts under ``scripts/test`` to evaluate the performance of **any RQA system that can be loaded with our** ``SimpleRQA`` **class**, which includes most of huggingface and OpenAI models.

For example, to evaluate a **retriever**:

.. code-block:: bash

    python scripts/test/test_retriever.py \
    --embedding_model_name_or_path intfloat/e5-base-v2 \
    --document_path <path/to/documents/filename.pkl> \
    --index_path <path/to/index> \
    --eval_data_path <path/to/test/data/filename.jsonl> \
    --output_dir <path/to/save/outputs>

this will load the model ``intfloat/e5-base-v2`` model (from huggingface in this case), read the documents database from ``document_path`` (and index them if ``index_path`` is empty), and evaluate the model on the test data from ``eval_data_path``. The model outputs and scores will be saved in ``output_dir``.



To evaluate **a retriever and a QA model end-to-end**:

.. code-block:: bash

    python scripts/test/test_e2e.py \
    --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
    --embedding_model_name_or_path intfloat/e5-base-v2 \
    --document_path <path/to/documents/filename.pkl> \
    --index_path <path/to/index> \
    --eval_data_path <path/to/test/data/filename.jsonl> \
    --output_dir <path/to/save/outputs>

this will load the embedding model ``intfloat/e5-base-v2`` and the QA model ``lmsys/vicuna-7b-v1.5`` (both from huggingface), read the documents database and indices, and evaluate the system as an ``SimpleRQA`` object on the test data from ``eval_data_path``. The system outputs and scores will be saved in ``output_dir``.


.. note::
    The evaluation scripts are currently designed to be run on a single machine. If you want to use a large model with a multi-GPU setting, you may use :ref:`serving-acc-frameworks` and **provide a model API endpoint** to ``qa_model_name_or_path``.


For more details on automatic evaluation, please refer to :ref:`evaluation-retriever` and :ref:`evaluation-e2e`.


.. toctree::
    :maxdepth: 1
    :caption: Evaluation
    :hidden:

    evaluation/eval_retriever.rst
    evaluation/eval_e2e.rst