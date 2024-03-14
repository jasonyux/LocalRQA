.. _train-main:

Training
========

Most examples we showed in this documentation used pre-trained models as-is. For example, we often used ``lmsys/vicuna-7b-v1.5`` and ``mistralai/Mistral-7B-Instruct-v0.2`` for QA models, and ``intfloat/e5-base-v2`` for our dense embedding models. However, in practice, **finetuning** these models with task-specific data often leads to **much better performance** (Karpukhin et al., 2020, Izacard et al., 2021; Wang et al., 2022).

To this end, we provide various training algorithms for both dense retrieval and generative models. This (non-exhaustive and growing) collection of algorithms comes from the latest research in the field, and we provide a simple interface to train these models with your own data.

+-----------+-----------------+--------------------------+-------------------------+
| Algorithm |  Target Model   |      Training Data       |         Section         |
+===========+=================+==========================+=========================+
| CTL       | Encoder         | (q, p, hard negative p†) |                         |
+-----------+-----------------+--------------------------+-------------------------+
| DCA       | Encoder         | (q, p†)                  |                         |
+-----------+-----------------+--------------------------+-------------------------+
| RPG       | Encoder         | (q, p†, a)               |                         |
+-----------+-----------------+--------------------------+-------------------------+
| SFT       | Decoder         | (h†, q, p, a)            | :ref:`training-gen-sft` |
+-----------+-----------------+--------------------------+-------------------------+
| SwR       | Decoder         | (h†, q, p, a)            | :ref:`training-gen-swr` |
+-----------+-----------------+--------------------------+-------------------------+
| FiD       | Encoder-Decoder | (h†, q, p, a)            | :ref:`training-gen-fid` |
+-----------+-----------------+--------------------------+-------------------------+


where "h" stands for **chat history**, "q" stands for a **question**, "p" stands for a (gold) **passage**, and "a" stands for a (gold) **answer**. "†" indicates this element is optional. We implement our trainers based on the `Trainer <https://huggingface.co/docs/transformers/main_classes/trainer>`_ class from the ``transformers`` library. This means that most of the arguments are the same as if you are familiar with the ``transformers`` library.


.. note::
   Currently, our training scripts assume a single GPU setting, since our evaluation loop assumes all models are loaded on a single GPU. **To avoid unexpected behavior with multi-GPU training**, we suggest setting ``--evaluation_strategy=no`` with our current training scripts.


**References:**

* Vladimir Karpukhin, Barlas O ̆guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense passage retrieval for open-domain question answering.
* Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning.
* Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly supervised contrastive pre-training.


.. toctree::
   :maxdepth: 5
   :caption: Training
   :hidden:

   training/retriever.rst
   training/generator.rst