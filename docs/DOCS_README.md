# Generate documentations using `sphinx`:

1. make sure you installed sphinx. This includes:
    ```bash
    pip install sphinx sphinx-book-theme myst_nb m2r2 sphinx-automodapi==0.16.0 autodoc_pydantic
    ```
2. go to the root project directory and update the API documentations with auto generated docstrings:
    ```bash
    /workspace/OpenRQA$ sphinx-apidoc -a --tocfile index -o docs/api_reference local_rqa/
    sphinx-apidoc --implicit-namespaces --tocfile index -o docs/api_reference local_rqa/
    ```
    this will generate `.rst` stuff under `docs/api_reference`.
3. go to the `docs` folder and generate the actual HTMLs!
    ```
    /workspace/OpenRQA/docs$ make html
    ```

4. start the server simply using:
    ```
    /workspace/OpenRQA/docs$ python -m http.server 5500
    ```

    then visit: `http://localhost:5500/_build/html/index.html` to see the documentations!