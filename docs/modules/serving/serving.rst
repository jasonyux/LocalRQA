Serving
=======
LocalRQA offers a simple way for researchers to publicize their RQA systems and also collect human feedback to further improve their systems. We implement: (1) efficient retrieval and LLM inference acceleration methods to improve users' experience during interactive chats, and (2) user interfaces designed for users to interact with their systems or for gathering human evaluations.

Default
-------
By default, all server logs will go under ``logs`` folder. Make sure this folder exists before running the commands below.

1. ``python open_rqa/serve/controller.py``

2. This is where you launch your customized RQA Pipeline(s)::

    python open_rqa/serve/model_worker.py \
    --document_path path/to/document.pkl \
    --index_path path/to/document/index \
    --embedding_model_name_or_path intfloat/e5-base-v2 \
    --qa_model_name_or_path lmsys/vicuna-7b-v1.5 \
    --model_id simple_rqa

3. To do a quick test to see if the above is working, try ``python open_rqa/serve/test_message.py --model_id simple_rqa``

4. Launch your demo page!
::

    python open_rqa/serve/gradio_web_server.py \
    --model_id simple_rqa \
    --example "What is DBFS? What can it do?" \
    --example "What is INVALID_ARRAY_INDEX?"


where the ``--model_id simple_rqa`` is to let the controller know which model this demo page is for, and the ``--example`` are the example questions that will be shown on the demo page.


Serving with Acceleration Framework
-----------------------------------

Here we provide an example using ``vLLM``. The procedure is very similar with using ``TGI``:

1. prepare and launch your ``vLLM`` server, hosting your generative model. For example::

    
    python -m vllm.entrypoints.api_server --model lmsys/vicuna-7b-v1.5 # This will by default host the model at `http://localhost:8000`.
    

2. Then all you have to do is to use ``--qa_model_name_or_path vllm::http://localhost:8000/generate`` instead of ``--qa_model_name_or_path lmsys/vicuna-7b-v1.5`` in the above section!

Currently our list of supported acceleration frameworks include ``vLLM``, ``SGLang``, and ``TGI``.


Serving Static Results for Evaluation
-------------------------------------

To evaluate the first 50 predictions from a prediction file (as done in our paper), run::

    python open_rqa/serve/gradio_static_server.py \
    --file_path path/to/save/output/test-predictions.jsonl \  # prediction file output from scripts/test/test_e2e.py
    --include_idx 1-50


we also allow a fine-grained customization (if needed) of which data to include in the evaluation. For example::

    python open_rqa/serve/gradio_static_server.py \
    --file_path path/to/save/output/test-predictions.jsonl \  # prediction file output from scripts/test/test_e2e.py
    --include_idx 1,2,5-12,14,25-63  # e.g., ask human to label data with idx [1,2] + [5,6,...,12] + [14] + [25,26,...,63]


and annotated data will be saved under ``logs/YY-MM-DD-HH-mm-annotations.jsonl``.