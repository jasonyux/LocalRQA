FROM nvcr.io/nvidia/pytorch:23.08-py3
ARG UNAME=docker
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN adduser $UNAME sudo

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER $UNAME
CMD /bin/bash
