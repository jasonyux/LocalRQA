.. _training-gen-swr:


SFT with a Frozen Retriever (SwR)
=================================

SwR finetunes a decoder using a combination of 1) **retrieved documents** from a frozen retriever, and 2) **ground-truth** <chat_history, question, passage, answer> pairs (for one turn QA, you can use ``chat_history=''``). Under the hood, this training script:

#. use the frozen retriever to retrieve ``k`` documents for each ``question`` in the training data.
#. augments the supporting passages to be ``passage_aug = passage + retrieved_passages``.
#. concatenates the ``chat_history``, ``question``, and ``passage_aug`` into a single input
#. trains the model to mimic the ground-truth ``answer`` using standard cross-entropy loss


Running SwR Trainer
-------------------

At a high level, SwR training requires:

* a **training, evaluation, and test dataset** of <question, passage, answer> pairs
* a **generative model** (e.g. ``mistralai/Mistral-7B-Instruct-v0.2``) to be trained
* an **embedding model** (e.g. ``intfloat/e5-base-v2``) used for training *AND* automatic E2E evaluation during training


Once you gathered these pieces, simply run:

.. code-block:: bash

    python scripts/train/qa_llm/train_w_fixed_retriever.py \
    --use_flash_attention true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    # other training hyperparameters omitted
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --assistant_prefix ASSISTANT \
    --user_prefix USER \
    --sep_user " " \
    --sep_sys "</s>" \
    --embedding_model intfloat/e5-base-v2 \
    --embedding_max_num_to_retrieve 3 \
    --output_dir model_checkpoints/my_SwR_qa_model \
    --train_file <example/train_w_qa.jsonl> \
    --eval_file <example/eval_w_qa.jsonl> \
    --test_file <example/test_w_qa.jsonl> \
    --full_dataset_file_path <example/documents.pkl> \
    --full_dataset_index_path <example/index>


for a full list of arguments, you can run ``python scripts/train/qa_llm/train_w_fixed_retriever.py -h``. In this example:

* ``--per_device_train_batch_size``, ``--model_name_or_path``, and other training arguments are from the HuggingFace `TrainingArguments <https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>`_ class. Since we implement our trainers from Huggingface's `Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_ class, it is compatible with most of the arguments there.
* ``--assistant_prefix`` and ``--user_prefix`` are the prefixes used to format the conversation history. This can be specific to the model you are training on (e.g., ``lmsys/vicuna-7b-v1.5``)
* ``--embedding_model`` is used to perform retrieval during training and evaluation.
* ``--embedding_max_num_to_retrieve`` dictates the size of ``passage_aug`` during training. In practice, we bound ``len(passage_aug) = embedding_max_num_to_retrieve + 1``.
* ``--output_dir`` is the directory where the trained model, training history, and evaluation results will be saved
* ``--train_file``, ``--eval_file``, and ``--test_file`` are the paths to the training, evaluation, and test datasets. See :ref:`data-description` for more details on the format of these files.
* ``--full_dataset_file_path`` and ``--full_dataset_index_path`` are the paths to the documents and their indices. This is used by ``eval_embedding_model`` to perform retrieval during evaluation. See :ref:`data-description` for more details on the format of these files.


.. note::
    For complete examples (e.g., obtaining files like ``<example/train_w_qa.jsonl>`` or other training hyperparameters), you can use :ref:`use-case-databricks` and :ref:`use-case-faire` as references.