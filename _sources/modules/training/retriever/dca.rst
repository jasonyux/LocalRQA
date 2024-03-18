.. _training-ret-dca:

Distill from Cross-Attention scores (DCA)
=========================================

DCA finetunes an embedding model by leveraging attention scores of a reader model to obtain synthetic labels for the retriever It often used for the task of learning information retrieval systems from unsupervised data (Izacard et al., 2020a). Under the hood, this training script:

#. Obtain the reader cross-attention score
#. Update the retrieval model parameters by minimizing the KL divergence between the similarity score of question *q* and passage *p* and the cross-attention score


Running DCA Trainer
-------------------

At a high level, DCA training requires:

* a **training, evaluation, and test dataset** 
* an **embedding model** (e.g. ``facebook/contriever-msmarco``) to be trained
* a **encode-decoder reader model** (e.g. ``google/flan-t5-xl``) used to generate cross-attention scores

Here is the sample of dataset. On top of the <question, passage> pairs, also need to prepare the "ctxs", which indicates the initial list of passages for each question. Different options can be considered to prepare that, such as DPR or BM25.

.. code-block:: bash

    [
        {
            "id": 0,
            "question": "",
            "gold_docs":[
                {
                    "page_content": "",
                    "fmt_content": "",
                    "metadata":{}
                }
            ],
            "ctxs":[
                {
                    "text": "",
                    "title": "",
                    "score": float (optional)
                }
            ]
        }
    ]

Once you gathered these pieces, simply run:

.. code-block:: bash

    python scripts/train/retriever/train_fid_retriever.py \
    --full_dataset_file_path <example/documents.pkl> \
    --train_file <example/train_w_q_fid.json> \
    --eval_file <example/eval_w_q_fid.json> \
    --model_name_or_path facebook/contriever-msmarco \
    --reader_model_path google/flan-t5-xl \
    --reader_temperature 0.1 \
    --with_score False \
    --projection True \
    --n_context 50 \
    --learning_rate 1e-5 \
    --reader_batch_size 4 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --metric_for_best_model eval_retr/document_recall/recall4 \
    --output_dir model_checkpoints/my_DCA_ret_model


for a full list of arguments, you can run ``python scripts/train/retriever/train_fid_retriever.py -h``. In this example:

* ``--per_device_train_batch_size``, ``--model_name_or_path``, and other training arguments are from the HuggingFace `TrainingArguments <https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>`_ class. Since we implement our trainers from Huggingface's `Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_ class, it is compatible with most of the arguments there.
* ``--output_dir`` is the directory where the trained model, training history, and evaluation results will be saved
* ``--full_dataset_file_path`` is the path to the documents. See :ref:`data-description` for more details on the format of these files.
* ``--n_context`` is the number of documents retrieved for each question. Equals to the length of ``ctxs``.


----

**References**:

* Gautier Izacard and Edouard Grave. 2020a. Distilling knowledge from reader to retriever for question answering.