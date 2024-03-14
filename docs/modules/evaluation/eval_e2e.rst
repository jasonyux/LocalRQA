.. _evaluation-e2e:

End-to-End Evaluation
=====================


To test the end-to-end performance of an RQA system, we provide an automatic evaluation script that measures:

* retrieval performance such as Recall@k;
* generation performance such as BLEU, ROUGE and GPT-4 Eval;
* end-to-end metrics such as runtime.

These are often often used in open-ended generation tasks such as machine translation and summarization. GPT-4 Eval is a recent method using GPT-4 (OpenAI, 2023) to evaluate the quality of model-generated responses (Liu et al., 2023; Zheng et al., 2023).


Running Evaluation
------------------

By default, our evaluation script ``scripts/test/test_e2e2.py`` is based on ``SimpleRQA`` class, which is compatible with most models from huggingface and OpenAI. To use this script, you will need 1) an **embedding model** and a **QA model**, 2) a **document/index database**, 3) a **test dataset**. For example:

.. code-block:: bash

    python scripts/test/test_e2e.py \
    --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
    --embedding_model_name_or_path intfloat/e5-base-v2 \
    --document_path <example/documents.pkl> \
    --index_path <example/index> \
    --eval_data_path <example/test_w_qa.jsonl> \
    --gen_gpt4eval false \
    --output_dir <example/output/dir>

this will output a JSONL file containing the evaluation results saved under ``<example/output/dir>``. The folder will have the following files after evaluation:

.. code-block:: bash
    
    <example/output/dir>
    ├── all_args.json           # arguments used for evaluation
    ├── score.json              # models performance
    ├── test-predictions.jsonl  # models predictions/outputs
    └─- test.log                # test logs


Note that:

- currently, we assume evaluation on *a single GPU*. If you want to use a large QA model with *multi-GPU*, please use :ref:`serving-acc-frameworks` to serve the model and then use the ``--qa_model_name_or_path`` argument to specify the endpoint.
- the ``document_path`` and ``index_path`` refer to the **document database** and the **indexed** folder. If ``index_path`` is empty, the script will also index the document database and save it to the specified path.
- by default, ``gen_gpt4eval`` is set to ``false``. If you want to use GPT-4 Eval, you should make sure you have configured ``export OPENAI_API_KEY=xxx`` and ``export OPENAI_ORGANIZATION=xxx``. Then, set ``--gen_gpt4eval true``.
- the ``test-predictions.jsonl`` file can then be **directly used with the** :ref:`serving-human-eval` **module**!
- for other available arguments, run ``python scripts/test/test_e2e.py -h``.


Customization
-------------

You can customize the behavior of the evaluation script by either modifying ``scripts/test/test_e2e2.py``, or using the evaluators in your own code. Our evaluation procedure simply consists of three steps:

.. code-block:: python

    from local_rqa.pipelines.retrieval_qa import SimpleRQA
    from local_rqa.evaluation.evaluator import E2EEvaluator, EvaluatorConfig

    def test(model_args: ModelArguments, test_args: TestArguments):
        ### 1. init rqa model. This returns a SimpleRQA object
        rqa_model = init_rqa_model(model_args, test_args.document_path, test_args.index_path)

        ### 2. define what metrics to use, and other configurations during evaluation
        eval_config = EvaluatorConfig(
            batch_size = test_args.batch_size,
            retr_latency = False,
            gen_f1 = True,
            gen_precision = True,
            gen_rouge = True,
            gen_latency = True,
            gen_gpt4eval = test_args.gen_gpt4eval,
            e2e_latency = True,
            ## eval model related configs
            assistant_prefix = model_args.assistant_prefix,
            user_prefix = model_args.user_prefix,
            sep_user = model_args.sep_user,
            sep_sys = model_args.sep_sys,
        )

        ### 3. load evaluation data, and run evaluation
        loaded_eval_data = load_eval_data(test_args.eval_data_path)
        evaluator = E2EEvaluator(
            config=eval_config,
            test_data=loaded_eval_data,
        )
        performance, predictions = evaluator.evaluate(rqa_model, prefix='test')
        # other code omitted
        return


.. note::

    Under the hood, the ``E2EEvaluator`` class takes in any RQA system that subclasses the ``RQAPipeline`` class (e.g., our ``SimpleRQA``). So if you wish to use a custom RQA system, you can first subclass ``RQAPipeline`` or even ``SimpleRQA``, and then simply pass it to the ``E2EEvaluator.evaluate``!


----

**References**:

* OpenAI. 2023. GPT-4. https://openai.com/gpt-4.

* Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

* Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023a. Judging llm-as-a-judge with mt-bench and chatbot arena.