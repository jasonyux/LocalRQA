.. _data-preparation:


Prepare RQA Data
=================

To use all components of LocalRQA effortlessly, you need to prepare two types of data: 1) a **document database**, and 2) a **collection of question-passage pairs or question-passage-answer triplets**. In general, we provide two methods to obtain these data:

* **Create your own data from scratch**: starting from your own data source (supported by ``langchain`` or ``llama-index``), you can create question-passage-answer using our sampling algorithm and an LLM.
* **Convert from existing QA datasets**: many existing QA datasets, such as Natural Questions, TriviaQA, and MS-Marco, contain question-passage-answer triplets. We can directly convert these datasets into the format used by LocalRQA.


Prepare Data from Scratch
-------------------------

* Get raw documents
* chunk them into a document database
* generate QA pairs



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