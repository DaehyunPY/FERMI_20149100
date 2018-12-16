FROM fedora:29
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

ENV LANG=en_US.UTF-8
RUN dnf update -y \
    && dnf install -y \
        which bzip2 git texlive nodejs \
        https://downloads.sourceforge.net/project/mscorefonts2/rpms/msttcore-fonts-installer-2.6-1.noarch.rpm \
    && dnf clean all

WORKDIR /root
ENV PATH /opt/anaconda/bin:$PATH
ADD https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh anaconda.sh
RUN bash anaconda.sh -bp /opt/anaconda \
    && rm anaconda.sh

WORKDIR /app
ENV PIPENV_VENV_IN_PROJECT=true \
    PATH=/app/.venv/bin:$PATH \
    PYSPARK_PYTHON=/app/.venv/bin/python
COPY Pipfile /app/
RUN pip install pipenv \
    && pipenv --python /opt/anaconda/bin/python --site-packages \
    && pipenv install \
    && pipenv install --dev \
    && python -m jupyter labextension install \
        @jupyterlab/toc \
        @jupyter-widgets/jupyterlab-manager \
        jupyterlab_bokeh \
    && rm -fr /root/.cache
RUN chmod a+wx /app

EXPOSE 8888
CMD [ "python" ]
