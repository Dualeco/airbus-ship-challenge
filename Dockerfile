FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 \
    python3-pip

RUN useradd -ms /bin/bash/ jupyter

USER jupyter

WORKDIR /home/jupyter

ENV PATH "$PATH:/home/jupyter/.local/bin"

COPY airbus-ships.ipynb ./airbus-ships.ipynb

COPY requirements.txt ./requirements.txt

COPY airbus_ships_final-14-0.36.hdf5 ./airbus_ships_final-14-0.36.hdf5

COPY train.py ./train.py

COPY rle.py ./rle.py

COPY model.py ./model.py

COPY lrfind.py ./lrfind.py

COPY inference.py ./inference.py

COPY img_utils.py ./img_utils.py

COPY callbacks.py ./callbacks.py

COPY ./test_v2 ./test_v2

RUN pip3 install -r requirements.txt

ENTRYPOINT ["jupyter", "notebook", "--ip=*"]