.. _serving-acc-frameworks:

Inference Acceleration Frameworks
==================================

RQA systems need time to perform retrieval and generate a response. If you are using large models, this latency can be high. This may not be acceptable for interactive chat systems, as in our example in :ref:`serving-interactive-eval`. To address this, we provide support for multiple acceleration frameworks that can be used to reduce the latency of the RQA system. This includes:

To speed up retrieval:

* `FAISS <https://github.com/facebookresearch/faiss>`_


To speed up text generation:

* `Text Generation Inference <https://github.com/huggingface/text-generation-inference>`_
* `vLLM <https://github.com/vllm-project/vllm>`_
* `SGLang <https://github.com/sgl-project/sglang>`_


By default, our ``SimpleRQA`` class (in :ref:`quickstart` and in :ref:`serving-interactive-eval`) uses ``FAISS`` for retrieval and no acceleration framework for text generation. However, you can **easily drop in any of the above acceleration frameworks in a two-step process**:

#. host your model using the acceleration framework of your choice, for instance, ``vLLM``:

   .. code-block:: bash

      python -m vllm.entrypoints.api_server --model lmsys/vicuna-7b-v1.5

   this should by default host the model at ``http://localhost:8000``.


#. Change the ``--qa_model_name_or_path`` argument to ``<framework-name>::<url>/generate``


Accelerating SimpleRQA
-----------------------

The ``SimpleRQA`` class is used in many contexts, such as during evaluation and model serving. Since the procedures are very similar for each framework, we will use ``SGLang`` in this example:

#. Use ``SGLang`` to host your generative model:

   .. code-block:: bash

      python -m sglang.launch_server --model-path lmsys/vicuna-7b-v1.5 --port 30000

#. Change the ``--qa_model_name_or_path`` argument to ``<framework-name>::<url>/generate``. For example, when you are evaluating your model:

   .. code-block:: bash
    
      python scripts/test/test_e2e.py \
      # --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
      --qa_model_name_or_path sglang::http://localhost:3000/generate
      --embedding_model_name_or_path intfloat/e5-base-v2 \
      --document_path <path/to/save/dir/filename.pkl> \
      --index_path <path/to/save/index> \
      --eval_data_path <path/to/test_w_qa.jsonl> \
      --output_dir <path/to/save/result/dir>



Accelerating Model Serving
-----------------------------

You can also use this ``<framework-name>::<url>/generate`` in our serving scripts, such as in :ref:`serving-interactive-eval`. Since the procedures are very similar for each framework, we will use ``vLLM`` as an example:

#. Use ``vLLM`` to host your generative model:

   .. code-block:: bash

      python -m vllm.entrypoints.api_server --model lmsys/vicuna-7b-v1.5

   this should by default host the model at ``http://localhost:8000``.

#. Change the ``--qa_model_name_or_path`` argument to ``<framework-name>::<url>/generate``:

   .. code-block:: bash
    
      export CUDA_VISIBLE_DEVICES=0
      python local_rqa/serve/model_worker.py \
      --document_path path/to/documents \
      --index_path path/to/index \
      --embedding_model_name_or_path intfloat/e5-base-v2 \
      # --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
      --qa_model_name_or_path vllm::http://localhost:8000/generate \
      --model_id simple_rqa