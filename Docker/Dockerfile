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
RUN curl -o anaconda.sh https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh \
    && bash anaconda.sh -bp /opt/anaconda \
    && rm anaconda.sh

WORKDIR /app
RUN pip install \
        altair \
        autopep8 \
        importlib_resources \
        lmfit \
        pyabel \
        git+https://github.com/DaehyunPY/vmitools.git \
    && rm -fr /root/.cache
RUN chmod a+wx /app

CMD [ "python" ]
