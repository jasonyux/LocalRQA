.. _serving-human-eval:

Static Human Evaluation
====================

A static evaluation page allows real users to evaluate your models' predictions. This can be used to augment your automatic evaluation results (see :ref:`evaluation-main`), or to provide a more detailed analysis of your models' performance.

At a high level, **all you need to perpare is a JSONL prediction file** (e.g., automatically produced by our evaluation scripts in :ref:`evaluation-main`). Then, you can simply run the `gradio_static_server.py` to launch a web server that allows other users to evaluate the quality of the pre-generated responses. Evaluation results will be automatically saved to the `/log` directory under the root of the project.


.. figure:: /_static/serving/human_eval_ui.png
   :scale: 60 %
   :alt: Static Evaluation UI

   Static Evaluation


In more details:

#. prepare