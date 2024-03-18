.. _use-case-databricks:

Databricks
==========

About Databricks:

    With origins in academia and the open source community, Databricks was founded in 2013 by the original creators of Apache Spark™, Delta Lake and MLflow. As the world's first and only lakehouse platform in the cloud, Databricks combines the best of data warehouses and data lakes to offer an open and unified platform for data and AI.

To find out more about Databricks, you can visit `databricks.com <https://www.databricks.com/>`_.


.. figure:: /_static/usecases/databricks-logo.png
    :align: center
    :width: 60 %
    :alt: Databricks

    Image credit: https://www.databricks.com/


**In this tutorial**, we provide an end-to-end example of using LocalRQA to:

#. Prepare RQA data using technical documents obtained from Databricks's websites
#. Train a retriever model
#. Train a generator model
#. Use automatic metrics to measure the RQA system's performance
#. Deploy the RQA system for human evaluation/interactive free chat


Prepare Data
------------

Thanks to Databricks' technical team, we first obtained pre-processed text data from documents on the `docs.databricks.com <https://docs.databricks.com>`_ and `kb.databricks.com <https://kb.databricks.com>`_ domain. You can find these data under ``<example/databricks/raw>`` directory. These data include API references and technical tutorials for using Databricks' products:


.. code-block:: json
    
    // docs.jsonl
    [
        {
            "description": "",
            "text": "---\ndescription: Learn about delegated authentication to third party services managed by <Databricks>, like the Ideas Portal ...",
            "source": "https://docs.databricks.com/delegated-auth.html",
            "timestamp": 0,
            "next": "",
            "full_text": "---\ndescription: Learn about delegated authentication to third party services managed by <Databricks>, like the Ideas Portal ..."
        },
        {
            "description": "",
            "text": "---\ndescription: How-to guidance and reference information for data analysts, data scientists, and data engineers working in the ...",
            "source": "https://docs.databricks.com/index.html",
            "timestamp": 0,
            "next": "",
            "full_text": "---\ndescription: How-to guidance and reference information for data analysts, data scientists, and data engineers working ..."
        },
        ...
    ]


Chunking and Formatting
~~~~~~~~~~~~~~~~~~~~~~~

The first step is to chunk these documents and format them into ``local_rqa.schema.document.Document`` objects. This will essentially convert these raw data into a single **document database** that can be used for all subsequent training and evaluation steps. For JSONL files, this can be done using the following command:

.. code-block:: bash

    python scripts/data/process_docs.py \
    --document_path <example/databricks/raw/docs.jsonl> \
    --chunk_size 400 \
    --chunk_overlap_size 50 \
    --save_dir <example/databricks> \
    --save_name <documents>

This will read in the JSON file, chunk all texts into documents of maximum token length 400, and save the resulting document database into ``<example/databricks/documents.pkl>``.

*Alternatively*, you could also customize its behavior using ``langchain`` and our ``LangChainTextLoader`` (or if you use ``llama-index``, we also have ``LlamaIndexTextLoader``). Under the hood, this process of loading data and chunking them is done by:

.. code-block:: python

    from langchain.document_loaders import JSONLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from local_rqa.text_loaders.langchain_text_loader import LangChainTextLoader

    # other code omitted
    def main(args):
        loader_func, splitter_func = JSONLoader, RecursiveCharacterTextSplitter.from_huggingface_tokenizer

        ## configure how to load the data
        loader_parameters = {
            'file_path': '<example/databricks/raw/docs.jsonl>,
            'jq_schema': '.',
            'content_key': 'text',
            'json_lines': True,
            'metadata_func': metadata_func
        }
        
        ## configure how to chunk each piece of text
        ## RecursiveCharacterTextSplitter requires a tokenizer. As an example we can use one from hugginface
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        splitter_parameters = {
            'tokenizer': tokenizer,
            'chunk_size': 400,
            'chunk_overlap': 50
        }

        ## actually load and chunk the data
        kwargs = {"loader_params": loader_parameters, "splitter_params": splitter_parameters}
        documents = LangChainTextLoader(
            save_folder="<example/databricks>",
            save_filename="<documents>",
            loader_func=loader_func,
            splitter_func=splitter_func
        ).load_data(**kwargs)
        return documents  ## a document database


Both of the above should result in a document database of 11,136 passages. The content inside ``<example/databricks/documents.pkl>`` looks like:

.. code-block:: python

    [
        Document(page_content="---\ndescription: Learn about delegated ...", fmt_content="...", metadata={...}),
        Document(page_content="---\ndescription: How-to guidance and reference ...", fmt_content="...", metadata={...}),
        ...
    ]


Generating QA
~~~~~~~~~~~~~~

The above only gives us a document database. To train a QA system, we need question-passage-answer triplets. LocalRQA provides the following three-step method to generate QA pairs from a document database:

#. select a set of gold passages from the document database
#. for each gold passage, prompt an LLM to generate a question
#. for each gold passage and question, prompt an LLM to generate an answer


**Generate Questions**

Step 1 and step 2 are done together by the following command:

.. code-block:: bash

    python scripts/data/doc_to_q_databricks.py \
    -mode all \
    -document_path <example/databricks/documents.pkl> \
    --prompt_model gpt-3.5-turbo \
    --num_hard_negs_per_doc 2 \
    --num_train_data 1200 \  # use a small number to test if it works first
    --num_eval_test_data 150 \  # use a small number to test if it works first
    --save_dir <example/databricks>


This script first samples "(gold passage, hard negative passage 1, hard negative passage 2)" from the document database, and then prompts OpenAI's GPT-3.5-turbo to generate two questions for each gold passage. Then, 1200 will go to ``train_w_q.jsonl``, and 150 will be split to become  ``eval_w_q.jsonl``, ``test_w_q.jsonl`` under ``<example/databricks>`` folder.


If you peeked into ``scripts/data/doc_to_q_databricks.py``, you will notice that it simply does:

.. code-block:: python

    from scripts.data.doc_to_q import *

    DATABRICKS_DOC2Q_PROMPT = ...

    def databricks_filter_fn(doc: Document):
        # decides if we should keep this doc for question generation or not
        

    def main(args: argparse.Namespace):
        """to customize how (doc, q) pairs would be created, simply copy this function over and modify the "# customizable" parts
        """
        random.seed(0)
        if args.mode in ["init_eval_dset", "all"]:
            documents_dataset = create_positive_n_negative_examples(
                args=args,
                filter_fn=databricks_filter_fn  # customized
            )
            logger.info(f"Created {len(documents_dataset)} <gold document, hard negative documents> pairs.")
        if args.mode in ["create_eval_dset", "all"]:
            eval_dataset, test_dataset = create_heldout_test_dset(
                args,
                doc2q_prompt=DATABRICKS_DOC2Q_PROMPT  # customized
            )
            logger.info(f"Number of eval samples: {len(eval_dataset)}")
            logger.info(f"Number of test samples: {len(test_dataset)}")
        if args.mode in ["create_train_dset", "all"]:
            train_dataset = create_train_dset(
                args,
                doc2q_prompt=DATABRICKS_DOC2Q_PROMPT  # customized
            )
            logger.info(f"Number of train samples: {len(train_dataset)}")
        return


This means you can customize the prompt used for question generation, and the filter function used to decide which documents to use for question generation. In the end, the content of ``train_w_q.jsonl`` should look like:

.. code-block:: jsonl

    [
        {"chat_history": [], "questions": ["How do I authenticate ...", "What is ..."], "gold_docs": [...], "hard_neg_docs": [...]},
        {"chat_history": [], "questions": ["...", "..."], "gold_docs": [...], "hard_neg_docs": [...]},
        ...
    ]


**Generate Answers**

Finally, given a question and a gold passage, answer generation is straightforward. We can prompt another LLM to provide an answer given the question and the gold passage. This can be done using:

.. code-block:: bash

    python scripts/data/doc_q_to_a_databricks.py \
    --prompt_model gpt-4-1106-preview \
    --dataset_w_q <example/databricks/train_w_q.jsonl> \  # generated by the previous step
    --save_name train_w_qa.jsonl \
    --save_dir <example/databricks> \
    --end_data_idx 4  # a small number first to test if the answers are satisfactory


This will prompt OpenAI's GPT-4-turbo (``gpt-4-1106-preview``) to generate answers for each question and gold passage pair. The result data is saved to ``train_w_qa.jsonl`` under ``<example/databricks>`` folder. The content of ``train_w_qa.jsonl`` will look like:


.. code-block:: jsonl

    [
        {"chat_history": [], "question": "How do I authenticate ...", "gold_docs": [...], "hard_neg_docs": [...], "gold_answer": "..."},
        {"chat_history": [], "question": "What is ...", "gold_docs": [...], "hard_neg_docs": [...], "gold_answer": "..."},
        ...
    ]


To obtain ``eval_w_qa.jsonl`` and ``test_w_qa.jsonl``, you can simply replace the ``--dataset_w_q`` argument with ``<example/databricks/eval_w_q.jsonl>`` and ``<example/databricks/test_w_q.jsonl>`` respectively.


Train a Retriever
------------------

Now we have all the data we need. We can first use it to fine-tune a retriever model. In this example, we will use ``intfloat/e5-base-v2`` as the base model:


.. note::

    In this tutorial, we are using one A100 80GB GPU to train all of our models. You may want to adjust hyperparameters such as batch size and gradient accumulation steps if you are using a different setup.


.. code-block:: bash

    python scripts/train/retriever/train_ctl_retriever.py \
    --pooling_type mean \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 \
    --hard_neg_ratio 0.05 \
    --metric_for_best_model eval_retr/document_recall/recall4 \
    --model_name_or_path intfloat/e5-base-v2 \
    --max_steps 150 \
    --eval_steps 5 \
    --save_steps 5 \
    --logging_steps 1 \
    --temperature 1 \
    --output_dir <example/ctl/model/dir> \
    --train_file <example/faire/train_w_q.jsonl> \
    --eval_file <example/faire/eval_w_q.jsonl> \
    --test_file <example/faire/test_w_q.jsonl> \
    --full_dataset_file_path <example/databricks/documents.pkl>

This will finetune the model using the :ref:`training-ret-ctl` algorithm, log the training process to ``wandb``, and save the model with highest Recall@4 score to ``<example/ctl/model/dir>``.


For more details on **other training algorithms we currently support**, please refer to :ref:`training-ret`.


Train a Generator
------------------

Next, we can also fine-tune a generative model using the same data (and optionally the retriever we just trained). In this example, we will use ``mistralai/Mistral-7B-Instruct-v0.2`` as the base model:


.. code-block:: bash

    python scripts/train/qa_llm/train_w_gt.py \
    --use_flash_attention true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --deepspeed scripts/train/ds_config.json \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --bf16 true \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --assistant_prefix [/INST] \
    --user_prefix "<s>[INST]" \
    --sep_user " " \
    --sep_sys "</s>" \
    --eval_embedding_model <example/ctl/model/dir> \
    --logging_steps 10 \
    --eval_steps 30 \
    --save_steps 30 \
    --output_dir <example/sft/model/dir> \
    --run_group <example_wandb_run_group_name> \
    --train_file <example/databricks/train_w_qa.jsonl> \
    --eval_file <example/databricks/eval_w_qa.jsonl> \
    --test_file <example/databricks/test_w_qa.jsonl> \
    --full_dataset_file_path <example/databricks/documents.pkl> \
    --full_dataset_index_path <example/databricks/ctl/index>


This will finetune the model using the :ref:`training-gen-sft` algorithm for 2 epochs, log the training process to ``wandb``, and save the model to ``<example/sft/model/dir>``. During training, it will also perform :ref:`evaluation-e2e` with ``eval_embedding_model`` using the ``full_dataset_file_path`` and ``full_dataset_index_path``.


For more details on **other training algorithms we currently support**, please refer to :ref:`training-gen`.


Automatic Evaluation
--------------------

By default, our training scripts will perform automatic evaluation during training. However, there are circumstances where you may want to manually evaluate your model, for example, to swap in other embedding models for E2E evaluation. To this end, we provide standalone scripts for both retriever and generator evaluation.


To evaluate your retriever, for instance ``<example/ctl/model/dir>``:

.. code-block:: bash

    python scripts/test/test_retriever.py \
    --embedding_model_name_or_path <example/ctl/model/dir/checkpoint-xxx> \
    --document_path <example/databricks/documents.pkl>\
    --index_path <example/databricks/ctl/index> \
    --eval_data_path <example/databricks/test_w_q.jsonl> \
    --output_dir <example/retriever>

By default, this will evaluate ``embedding_model_name_or_path`` model's Recall@1, Recall@4 and runtime latency metrics using test data in ``<example/databricks/test_w_q.jsonl>``. The result will be saved as ``<example/retriever/test-predictions.jsonl>``. To enble **nDCG** metric, set *retr_ndcg = True* in setting ``EvaluatorConfig``.


To evaluate the generator, for instance ``<example/sft/model/dir>``:

.. code-block:: bash

    python scripts/test/test_e2e.py \
    --qa_model_name_or_path <example/sft/model/dir/checkpoint-xxx> \
    --assistant_prefix [/INST] \
    --user_prefix "<s>[INST]" \
    --sep_user " " \
    --sep_sys "</s>" \
    --embedding_model_name_or_path <example/ctl/model/dir/checkpoint-xxx> \
    --document_path <example/databricks/documents.pkl> \
    --index_path <example/databricks/ctl/index> \
    --eval_data_path <example/databricks/test_w_qa.jsonl> \
    --output_dir <example/e2e>


This will treat the ``qa_model_name_or_path`` and the ``embedding_model_name_or_path`` as an RQA system, and evaluate end-to-end using test data in ``<example/databricks/test_w_qa.jsonl>``. The result will be saved as ``<example/e2e/test-predictions.jsonl>``.


.. note::

    The evaluation command above does **not** perform GPT-4 based evaluation. To enable that option, you can pass in ``--gen_gpt4eval true``. We note that this will incur additional costs, and is only recommended for final evaluation.

    For more details on end-to-end evaluation, please refer to :ref:`evaluation-e2e`.


As a baseline, you could also test the performance of GPT-4-turbo with text-ada-002:


.. code-block:: bash

    python scripts/test/test_e2e.py \
    --qa_model_name_or_path gpt-4-1106-preview \
    --embedding_model_name_or_path text-embedding-ada-002 \
    --document_path <example/databricks/documents.pkl> \
    --index_path <example/databricks/openai/index> \
    --eval_data_path <example/databricks/test_w_qa.jsonl> \
    --output_dir <example/openai/e2e>


Deploy the RQA system
---------------------

If you are satisfied with your current RQA system, you can deploy it for human evaluation or interactive free chat. Human evaluation results can be used to validate performance beyond automatic evaluation, and feedback from interactive free chat can be used to further improve the RQA system.


**To deploy the RQA system above for human evaluation**, you can do:

.. code-block:: bash

    python local_rqa/serve/gradio_static_server.py \
    --file_path <example/e2e/test-predictions.jsonl> \
    --include_idx 1-50


This will launch a Gradio server at port ``7861`` and display the first 50 examples. You can access it by visiting ``http://localhost:7861`` in your browser, or share the link with others for human evaluation. For more details on our static human evaluation server, please refer to :ref:`serving-human-eval`.


**Deploying the system for interactive free chat** is more complicated, as it requires hosting the model and managing asynchronous requests. We provide a quick example below. You may want to refer to :ref:`serving-interactive-eval` for more details.

#. start a controller with ``python local_rqa/serve/controller.py``.
#. start your model worker with

   .. code-block:: bash

      python local_rqa/serve/model_worker.py \
      --document_path <example/databricks/documents.pkl> \
      --index_path <example/databricks/openai/index> \
      --embedding_model_name_or_path <example/ctl/model/dir/checkpoint-xxx> \
      --qa_model_name_or_path <example/sft/model/dir/checkpoint-xxx> \
      --model_id simple_rqa

#. test if the model worker is alive: ``python local_rqa/serve/test_message.py --model_id simple_rqa``

#. finally. launch the web server with:

   .. code-block:: bash

      python local_rqa/serve/gradio_web_server.py \
      --port 28888 \
      --model_id simple_rqa \
      --example "What is DBFS? What can it do?" \
      --example "What is INVALID_ARRAY_INDEX?"


You are all set! To access this server, you can visit ``http://localhost:28888`` in your browser. By default, any server log will be saved to the ``logs/`` folder. You can then access this log folder for chat histories and users' feedback when chatting with your system!