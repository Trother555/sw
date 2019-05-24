FROM python:3.6

WORKDIR /src

COPY . /src

RUN apt update &&\
    apt-get update &&\
    apt-get -y install swig locales &&\
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen &&\
    locale-gen

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /src/JamSpell

RUN python setup.py install

WORKDIR /src

CMD ["python", "server.py"]