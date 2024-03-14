.. _evaluation-retriever:

Retriever Evaluation
====================


To test the retriever performance, we provide an automatic evaluation script that measures:

* Recall@k, such as Recall@1 and Recall@4;
* nDCG@k;
* end-to-end metrics such as runtime.

Recall and nDCG scores are often used in information retrieval benchmarks such as BEIR (Thakuret al., 2021) and MTEB (Muennighoff et al., 2022). Runtime is important for real-world applications.


Running Evaluation
------------------

By default, our evaluation script ``scripts/test/test_retriever.py`` loads the retriever based on the ``FaissRetriever`` class , which is compatible with most models from huggingface and OpenAI. To use this script, you will need 1) an **embedding model**, 2) a **document/index database**, 3) a **test dataset**. For example:

.. code-block:: bash

    python scripts/test/test_retriever.py \
    --embedding_model_name_or_path intfloat/e5-base-v2 \
    --document_path <example/documents.pkl> \
    --index_path <example/index> \
    --eval_data_path <example/test_w_q.jsonl> \
    --output_dir <example/output/dir>

this will output a JSONL file containing the evaluation results saved under ``<example/output/dir>``. The folder will have the following files after evaluation:

.. code-block:: bash
    
    <example/output/dir>
    ├── all_args.json           # arguments used for evaluation
    ├── score.json              # models performance
    ├── test-predictions.jsonl  # models predictions/outputs
    └─- test.log                # test logs


Note that:

- the ``document_path`` and ``index_path`` refer to the **document database** and the **indexed** folder. If ``index_path`` is empty, the script will also index the document database and save it to the specified path.
- Even though ``embedding_model_name_or_path`` can be any embedding model from Huggingface, but the model would be better saved in **SentenceTransformer**, because it will have the config file for the pooling method used to generate final embedding, otherwise it will use mean pooling by default.


Customization
-------------

You can customize the behavior of the evaluation script by either modifying ``scripts/test/test_retriever.py``, or using the evaluators in your own code. Our evaluation procedure simply consists of three steps:

.. code-block:: python

    from local_rqa.retrievers.faiss_retriever import FaissRetriever
    from local_rqa.evaluation.evaluator import RetrieverEvaluator, EvaluatorConfig

    def test(model_args: ModelArguments, test_args: TestArguments):
        ### 1. init retriever model. This returns a FaissRetriever object
        retriever_model = init_retriever_model(model_args, test_args, documents)

        ### 2. define what metrics to use, and other configurations during evaluation
        eval_config = EvaluatorConfig(
            batch_size = test_args.batch_size
        )

        ### 3. load evaluation data, and run evaluation
        loaded_eval_data = load_eval_data(test_args.eval_data_path)
        evaluator = RetrieverEvaluator(
            config=eval_config,
            test_data=loaded_eval_data,
        )
        performance, predictions = evaluator.evaluate(retriever_model, prefix='test')
        # other code omitted
        return

----

**References**:

* Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021 BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

* Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2022. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316.
