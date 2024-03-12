# Generate documentations using `sphinx`:

1. make sure you installed sphinx. This includes:
    ```bash
    pip install sphinx sphinx-book-theme myst_nb m2r2 sphinx-automodapi==0.16.0 autodoc_pydantic
    ```
2. go to the root project directory and update the API documentations with auto generated docstrings:
    ```bash
    /workspace/LocalRQA$ sphinx-apidoc --implicit-namespaces --tocfile index -o docs/api_reference local_rqa/
    ```
    this will generate `.rst` stuff under `docs/api_reference`.
3. if you skipped the previous step (not always needed unless you updated your documentations), make sure the you have `_static` and `_templates` (empty folders) under your `docs`:
    ```bash
    /workspace/LocalRQA/docs$ tree -L 1
    ├── Makefile
    ├── _build
    ├── _static
    ├── _templates
    ├── (some other files and folders)
    ├── index.rst
    └── make.bat
    ```
    if not you can just manually create them.
4. go to the `docs` folder and generate the actual HTMLs!
    ```bash
    /workspace/LocalRQA/docs$ make html
    ```
5. start the server simply using:
    ```bash
    /workspace/LocalRQA/docs$ python -m http.server 5500
    ```
    then visit: `http://localhost:5500/_build/html/index.html` to see the documentations!


## How to add a custom page on the sidebar?

Under the hood, everything is easily configured under the `index.rst` home page. For instance:

```rst
Welcome to LocalRQA's documentation!
===================================

LocalRQA is an (some text omitted)

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/scripts.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_reference/local_rqa.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

```

those side sections are added by this sippet:

```rst

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/scripts.md
   getting_started/xxxx.md

```

so all you need to do is to add the corresponding `xxxx.md` files under `docs/getting_started/xxxx.md` and you are good to go!

Some useful references when writing custom pages:

- [adding cross-references](https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html#getting-started)
- [adding images](https://stackoverflow.com/questions/25866102/how-do-we-embed-images-in-sphinx-docs) (you might need to use `.rst` for this) 