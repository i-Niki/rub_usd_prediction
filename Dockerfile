FROM tensorflow/tensorflow
RUN python -m pip install --upgrade pip
COPY ./src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY ./src/final.py /main/final.py
WORKDIR main
CMD python final.py
