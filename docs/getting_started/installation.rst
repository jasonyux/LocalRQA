.. _installation:

Installation
============

You can either install the package from GitHub or use our pre-built Docker image.


From GitHub
-----------

First, clone our repository

.. code-block:: bash

    git clone https://github.com/jasonyux/LocalRQA
    cd LocalRQA


Then run

.. code-block:: bash

    pip install --upgrade pip
    pip install -e .


From Docker
-----------

.. code-block:: bash

    docker pull jasonyux/localrqa
    docker run -it jasonyux/localrqa


our code base is located at ``/workspace/LocalRQA``.