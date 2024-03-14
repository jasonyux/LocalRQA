.. _training-ret-rpg:

Distill from an LM's probability distribution (RPG)
===================================================

RPG finetunes an embedding model by using the LM's probability districution of the gold answer to provide supervision about which documents should be retrieved (Shi etal., 2023). Under the hood, this training script:

#. For each training step, use the current retriever retrieve ``k`` documents for each ``question`` in the training data.
#. Compute the retrieval likelihood for each retrieved documents
#. Compute the LM likelihood of the gold answer based on each retrieved documents
#. Update the retrieval model parameters by minimizing the KL divergence between the retrieval likelihood and the LM's probability distribution
#. Update the documents index with new retrieval model every **refresh_step**


Running RPG Trainer
-------------------

At a high level, RPG training requires:

* a **training, evaluation, and test dataset** of <question, passage, answer> pairs
* an **embedding model** (e.g. ``facebook/contriever-msmarco``) to be trained
* a **LM** (e.g. ``stabilityai/stablelm-zephyr-3b``) used to score language distribution

Once you gathered these pieces, simply run:

.. code-block:: bash

    python scripts/train/retriever/train_replug_retriever.py \
    --full_dataset_file_path <example/documents.pkl> \
    --train_file <example/train_w_qa.jsonl> \
    --eval_file <example/eval_w_qa.jsonl> \
    --model_name_or_path facebook/contriever-msmarco \
    --lm_model_path stabilityai/stablelm-zephyr-3b \
    --refresh_step 10 \
    --text_maxlength 512 \
    --lm_temperature 0.1 \
    --retrieve_temperature 0.1 \
    --num_docs 20 \
    --pooling_type mean \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --metric_for_best_model eval_retr/document_recall/recall4 \
    --output_dir model_checkpoints/my_RPG_ret_model


for a full list of arguments, you can run ``python scripts/train/retriever/train_replug_retriever.py -h``. In this example:

* ``--per_device_train_batch_size``, ``--model_name_or_path``, and other training arguments are from the HuggingFace `TrainingArguments <https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>`_ class. Since we implement our trainers from Huggingface's `Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_ class, it is compatible with most of the arguments there.
* ``--output_dir`` is the directory where the trained model, training history, and evaluation results will be saved
* ``--train_file`` and ``--eval_file`` are the paths to the training and evaluation datasets. See :ref:`data-description` for more details on the format of these files.
* ``--full_dataset_file_path`` is the path to the documents. See :ref:`data-description` for more details on the format of these files.
* ``--refresh_step`` is the hyperparameter that controls how often to update the document index.
* ``--num_docs`` is the number of retrieved documents ``k`` for each question during the training. Note that this is different from the number during inference, which is 4 by default.
* ``--lm_temperature`` and ``--retrieve_temperature`` are the two hyperparameters that controls the temperature of the softmax when calculaing LM and retrieval likelihood.


.. note::
    For complete examples (e.g., obtaining files like ``<example/train_w_qa.jsonl>`` or other training hyperparameters), you can use :ref:`use-case-databricks` and :ref:`use-case-faire` as references.


----

**References**:

* Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. Replug: Retrieval-augmented black-box language models.