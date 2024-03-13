.. _serving-human-eval:

Static Human Evaluation
====================

A static evaluation page allows real users to evaluate your models' predictions. This can be used to augment your automatic evaluation results (see :ref:`evaluation-main`), or to provide a more detailed analysis of your models' performance.

At a high level, **all you need to perpare is a JSONL prediction file** (e.g., automatically produced by our evaluation scripts in :ref:`evaluation-main`). Then, you can simply run the ``gradio_static_server.py`` to launch a web server that allows other users to evaluate the quality of the pre-generated responses. Evaluation results will be automatically saved to the ``log`` directory under the root of the project.


.. figure:: /_static/serving/human_eval_ui.png
   :scale: 60 %
   :alt: Static Evaluation UI

   Static Evaluation


In more details:

#. prepare a JSONL file which contains an input context (e.g., a ``question``), your model's retrieved documents and response (``gen_answer`` and ``retrieved_docs``, respectively), and optionally reference documents (``gold_docs``). For example:
      
   .. code-block:: json

      {"question": "How do I xxx?", "gold_docs": [{"fmt_content": ...}], "retrieved_docs": [{"fmt_content": ...}, ...], "generated_answer": "You can ..."}
      {"question": "What does xxx mean?", "gold_docs": [{"fmt_content": ...}], "retrieved_docs": [{"fmt_content": ...}, ...], "generated_answer": "xxx is ..."}
      ...

   Note that ``[{"fmt_content": ...}]`` is the dict version of ``local_rqa.schema.document.Document``.

   You can obtain such a JSONL file using the scripts in :ref:`evaluation-main`, or by manually preparing the file.

#. Run the ``gradio_static_server.py`` script to read the JSONL file and launch a web server. For example:

   .. code-block:: bash
         
         python open_rqa/serve/gradio_static_server.py \
         --file_path /path/to/test-predictions.jsonl \
         --include_idx 1-50  # (optional) display only the first 50 predictions

   The server will be launched at port 7861 by default.

#. You are all set! Once a user completed all the evaluations and click "Submit", these annotated data will be automatically saved under `logs/YY-MM-DD-HH-mm-annotations.jsonl`.