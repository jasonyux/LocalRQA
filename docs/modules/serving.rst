.. _serving-main:

Serving
=======

LocalRQA provides methods to 1) deploy your RQA system with a simple model backend, and 2) launch an interactive UI for human evaluation or free chat. This can be used to showcase your RQA system to the public, or to collect human feedback to further improve your system using techniques such as RLHF.


Launchers
---------


Currently, we provide methods to launch your RQA system in two different ways.

**Static evaluation**: users directly evaluate the quality of pre-generated responses (e.g., computed from a test set). See :ref:`serving-human-eval` for more details.

.. figure:: /_static/serving/human_eval_ui.png
   :scale: 60 %
   :alt: Static Evaluation UI

   Static Evaluation


**Interactive chat**: users can chat with a system and rate the correctness/helpfulness of each response. See :ref:`serving-interactive-eval` for more details.

.. figure:: /_static/serving/interactive_ui.png
   :scale: 60 %
   :alt: Interactive Chat UI

   Interactive Chat


Acceleration Frameworks
-------------------------

We also integrate with several inference acceleration frameworks to speed up model's retrieval/answer generation. Currently, we support:

For retrieval:

* FAISS (Johnson et al.,2019)


For text generation:

* Text Generation Inference (Huggingface, 2023)
* vLLM (Kwon et al., 2023)
* SGLang (Zheng et al., 2023)

See :ref:`serving-acc-frameworks` for more details of how to use them with the serving methods mentioned above.


.. toctree::
    :maxdepth: 1
    :caption: Serving
    :hidden:

    serving/human_eval.rst
    serving/interactive.rst
    serving/acc_frameworks.rst