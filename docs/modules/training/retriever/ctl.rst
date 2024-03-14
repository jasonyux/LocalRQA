.. _training-ret-ctl:

Contrastive Learning (CTL)
==========================

CTL finetunes an embedding model through in-batch contrastive learning. It employs gold <question, passage> pairs as positive examples and utilizes passages corresponding to other questions within the same batch as hard negatives. Under the hood, this training script:

#. flatten the dataset with one question for each passage
#. trains the model to distinguish between positive and negative data points using standard cross-entropy loss


Running CTL Trainer
-------------------

At a high level, CTL training requires:

* a **training, evaluation, and test dataset** of <question, passage> pairs
* an **embedding model** (e.g. ``intfloat/e5-base-v2``) to be trained

Once you gathered these pieces, simply run:

.. code-block:: bash

    python scripts/train/retriever/train_ctl_retriever.py \
    --full_dataset_file_path <example/documents.pkl> \
    --train_file <example/train_w_q.jsonl> \
    --eval_file <example/eval_w_q.jsonl> \
    --model_name_or_path intfloat/e5-base-v2 \
    --pooling_type mean \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 \
    --hard_neg_ratio 0.05 \
    --contrastive_loss inbatch_contrastive \
    --metric_for_best_model eval_retr/document_recall/recall4 \
    --output_dir model_checkpoints/my_CTL_ret_model


for a full list of arguments, you can run ``python scripts/train/retriever/train_ctl_retriever.py -h``. In this example:

* ``--per_device_train_batch_size``, ``--model_name_or_path``, and other training arguments are from the HuggingFace `TrainingArguments <https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>`_ class. Since we implement our trainers from Huggingface's `Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_ class, it is compatible with most of the arguments there.
* ``--output_dir`` is the directory where the trained model, training history, and evaluation results will be saved
* ``--train_file`` and ``--eval_file`` are the paths to the training and evaluation datasets. See :ref:`data-description` for more details on the format of these files.
* ``--full_dataset_file_path`` is the path to the documents. See :ref:`data-description` for more details on the format of these files.


.. note::
    For complete examples (e.g., obtaining files like ``<example/train_w_q.jsonl>`` or other training hyperparameters), you can use :ref:`use-case-databricks` and :ref:`use-case-faire` as references.