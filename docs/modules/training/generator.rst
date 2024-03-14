.. _training-gen:

Training a Generator
====================

Given *k* retrieved passages and a user query (and optionally a chat history), generative models produce an answer conditioned on all the inputs.
Besides improving retrievers, using better generative models can also improve an RQA system's performance. This is because **more capable generators can more effectively incorporate retrieved passages**, thereby providing answers that are more helpful and factual. 

To this end, we provide several training algorithms curated from the latest research in RQA:

* :ref:`training-gen-sft`: simply supervised finetunes a decoder model using **ground truth question-document-answer pairs**.
* :ref:`training-gen-swr`: supervised finetune a decoder with **a frozen retriever**
* :ref:`training-gen-fid`: finetunes an encoder-decoder with a frozen retriever using **fusion-in-decoder** training

At a high level, we provide ready-to-use training scripts for each algorithm above. These scripts allow you to specify the training data, model, and other hyperparameters in a single command line. For instance, with SFT training:


.. code-block:: bash

    python scripts/train/qa_llm/train_w_gt.py \
    --use_flash_attention true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    # other training hyperparameters omitted
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --assistant_prefix [/INST] \
    --user_prefix "<s>[INST]" \
    --sep_user " " \
    --sep_sys "</s>" \
    --eval_embedding_model intfloat/e5-base-v2 \
    --output_dir model_checkpoints/my_SFT_qa_model \
    --train_file <example/train_w_qa.jsonl> \
    --eval_file <example/eval_w_qa.jsonl> \
    --test_file <example/test_w_qa.jsonl> \
    --full_dataset_file_path <example/documents.pkl> \
    --full_dataset_index_path <example/index>


this will finetune ``mistralai/Mistral-7B-Instruct-v0.2`` using the training data from ``<example/train_w_qa.jsonl>``, and then save the model at ``model_checkpoints/my_SFT_qa_model``.

.. note::
    During training, our scripts will **also perform automatic E2E evaluations on the validation set**, i.e., ``<example/eval_w_qa.jsonl>``. This means that it will use ``eval_embedding_model`` as the retriever, and take ``full_dataset_file_path`` as the document database and ``full_dataset_index_path`` as the index. The evaluation results will be printed to the console and saved to the output directory.

    For more details on E2E evaluation, please refer to :ref:`evaluation-e2e`.


For more details on each training algorithm/script, please refer to their respective sections.


.. toctree::
   :maxdepth: 5
   :hidden:

   generator/sft.rst
   generator/swr.rst
   generator/fid.rst