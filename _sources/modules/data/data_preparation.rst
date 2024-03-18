.. _data-preparation:


Prepare RQA Data
=================

To use all components of LocalRQA effortlessly, you need to prepare two types of data: 1) a **document database**, and 2) a **collection of question-passage pairs or question-passage-answer triplets**. In general, we provide two methods to obtain these data:

* **Create your own data from scratch**: starting from your own data source (supported by ``langchain`` or ``llama-index``), you can create question-passage-answer using our sampling algorithm and an LLM.
* **Convert from existing QA datasets**: many existing QA datasets, such as Natural Questions, TriviaQA, and MS-Marco, contain question-passage-answer triplets. We can directly convert these datasets into the format used by LocalRQA.


Prepare Data from Scratch
-------------------------

The following steps present a high-level overview of how to prepare RQA data from scratch. For more concrete examples, please refer to our tutorial with :ref:`use-case-databricks` and :ref:`use-case-faire`!

Prepare Document Pickle File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LocalRQA supports integration with frameworks such as LangChain and LlamaIndex to easily ingest text data in various formats, such as JSON data, HTML data, data from Google Drive, etc.
In the :ref:`quickstart`, it shows an example of parsing and storing the website content with URLs as data source, here is another example of using JSONL format data as source.

Here is an sample of JSONL data format, where "text" key must be used to store the passage content. While, other keys (such as "source", "title", etc.) are optional:

.. code-block:: bash

    {
        "source": "<passage_source>", 
        "title": "<passage_title>",
        "text": "<passage_content>"
    }

Then run the following command,

.. code-block:: bash

    python scripts/data/process_docs.py \
    --document_path <example/docs.jsonl> \
    --model_name_or_path facebook/contriever-msmarco \
    --chunk_size 400 \
    --chunk_overlap_size 50 \
    --save_dir example \
    --save_name documents

It will:

* load jsonl file from ``<example/docs.jsonl>``
* split the text by **RecursiveCharacterTextSplitter.from_huggingface_tokenizer** based on the chunk size measured by **facebook/contriever-msmarco** tokenizer
* represent the documents with a list of Document object and save into ``<example/documents.pkl>`` file 

Here is a sample of pickle file:

.. code-block:: bash

    [
        Document(
            page_content='<passage_content>', 
            fmt_content='Source: <passage_source>\nTitle: <passage_title>\nContent:\n<passage_content>', 
            metadata={'source': '<passage_source>', 'seq_num': 1, 'title': '<passage_title>'}
        )
    ]



Generate Questions
~~~~~~~~~~~~~~~~~~

By using the prepared ``<example/documents.pkl>`` file, run the following script to generate questions with the help of LLM, such as gpt-3.5-turbo.

.. code-block:: bash

    python scripts/data/doc_to_q.py \
    -mode all \
    -document_path <example/documents.pkl> \
    --prompt_model gpt-3.5-turbo \
    --num_hard_negs_per_doc 2 \
    --num_train_data 600 \
    --num_eval_test_data 150 \
    --save_dir <example>

It will save **eval_w_q.jsonl, test_w_q.jsonl, train_w_q.jsonl** into the ``<example>`` directory.



Generate Answers
~~~~~~~~~~~~~~~~

Then, with the prepared **eval_w_q.jsonl, test_w_q.jsonl, train_w_q.jsonl** from the previous step, we will leverage on the LLM (such as OPENAI gpt-4-1106-preview model) again to generate the corresponding answer. Here is an example to generate **train_w_qa.jsonl** by using **train_w_q.jsonl** file.

.. code-block:: bash

    python scripts/data/doc_q_to_a.py \
    --prompt_model gpt-4-1106-preview \
    --dataset_w_q <example/train_w_q.jsonl> \
    --save_name train_w_qa.jsonl \
    --save_dir <example> \
    --end_data_idx 4  # a small number to test if it works





Convert from Existing Datasets
------------------------------

QA datasets such as Natural Questions (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and MS-Marco (Bajaj et al., 2018) implicitly contain question-passage-answer triplets. We provide scripts that can convert these datasets into the format used by LocalRQA.


For example, to convert the TriviaQA dataset:

.. code-block:: bash

    python scripts/data/load_hf_data.py \
    --dataset trivial_qa \
    --document_save_path <example/trivial_qa/documents> \
    --train_data_save_path <example/trivial_qa/training_data>


this will:

* store all the documents in the TrivialQA dataset as a document database under the ``example/trivial_qa/documents`` directory
* split the training, validation, and test data as ``train_w_qa.jsonl``, ``eval_w_qa.jsonl``, and ``test_w_qa.jsonl``, and store them under the ``example/trivial_qa/training_data`` directory.


These converted data contain **everything** you need to run LocalRQA's training, evaluation, and serving components. For all currently supported datasets, run ``python scripts/data/load_hf_data.py -h``.


**References**

* Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.
* Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics
* Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS-Marco: A human generated machine reading comprehension dataset.