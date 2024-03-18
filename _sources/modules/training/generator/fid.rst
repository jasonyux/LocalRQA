.. _training-gen-fid:


Fusion-in-Decoder Training (FiD)
================================

FiD finetunes an encoder-decoder using a combination of 1) **retrieved documents** from a frozen retriever, and 2) **ground-truth** <chat_history, question, passage, answer> pairs (for one turn QA, you can use ``chat_history=''``). Under the hood, this training script:

#. use the frozen retriever to retrieve ``k`` documents for each ``question`` in the training data.
#. augments the supporting passages to be ``passage_aug = passage + retrieved_passages``.
#. for each passage ``p`` in ``passage_aug``, concatenate with the chat history and question to form the input ``input_i = chat_history + question + p``.
#. encode each ``input_i`` using the encoder in **parallel**
#. concatenate the hidden states of the encoder and feed them into the encoder-decoder
#. trains the encoder-decoder model using standard cross-entropy loss on the ground-truth answer.

Visually:

.. figure:: /_static/training/fid.png
   :align: center
   :width: 800px
   :alt: Fusion-in-Decoder Training

   Architecture of the Fusion-in-Decoder method. (Izacard and Grave, 2020)


Running FiD Trainer
-------------------

At a high level, SwR training requires:

* a **training, evaluation, and test dataset** of <question, passage, answer> pairs
* an **encoder-decoder model** (e.g. ``lmsys/fastchat-t5-3b-v1.0``) to be trained
* an **embedding model** (e.g. ``intfloat/e5-base-v2``) used for training *AND* automatic E2E evaluation during training


.. code-block:: bash

    python scripts/train/qa_llm/train_w_gt_fid.py \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    # other training hyperparameters omitted
    --model_name_or_path lmsys/fastchat-t5-3b-v1.0 \
    --embedding_model intfloat/e5-base-v2 \
    --embedding_max_num_to_retrieve 3 \
    --output_dir model_checkpoints/my_SwR_qa_model \
    --train_file <example/train_w_qa.jsonl> \
    --eval_file <example/eval_w_qa.jsonl> \
    --test_file <example/test_w_qa.jsonl> \
    --full_dataset_file_path <example/documents.pkl> \
    --full_dataset_index_path <example/index>


for a full list of arguments, you can run ``python scripts/train/qa_llm/train_w_gt_fid.py -h``. In this example:

* ``--per_device_train_batch_size``, ``--model_name_or_path``, and other training arguments are from the HuggingFace `TrainingArguments <https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>`_ class. Since we implement our trainers from Huggingface's `Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_ class, it is compatible with most of the arguments there.
* ``--embedding_model`` is used to perform retrieval during training and evaluation.
* ``--embedding_max_num_to_retrieve`` dictates the size of ``passage_aug`` during training. In practice, we bound ``len(passage_aug) = embedding_max_num_to_retrieve + 1``.
* ``--output_dir`` is the directory where the trained model, training history, and evaluation results will be saved
* ``--train_file``, ``--eval_file``, and ``--test_file`` are the paths to the training, evaluation, and test datasets. See :ref:`data-description` for more details on the format of these files.
* ``--full_dataset_file_path`` and ``--full_dataset_index_path`` are the paths to the documents and their indices. This is used by ``eval_embedding_model`` to perform retrieval during evaluation. See :ref:`data-description` for more details on the format of these files.


.. note::
    For complete examples (e.g., obtaining files like ``<example/train_w_qa.jsonl>`` or other training hyperparameters), you can use :ref:`use-case-databricks` and :ref:`use-case-faire` as references.

----

**References**

* Gautier Izacard and Edouard Grave. 2020. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.