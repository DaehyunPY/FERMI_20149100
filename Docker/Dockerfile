FROM fedora:29
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

ENV LANG=en_US.UTF-8

WORKDIR /root
RUN dnf update -y \
    && dnf install -y \
        which bzip2 git texlive \
        https://downloads.sourceforge.net/project/mscorefonts2/rpms/msttcore-fonts-installer-2.6-1.noarch.rpm \
    && dnf clean all

# Avoid to use "Add" here to reduce the image size
RUN curl -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /root/miniconda.sh -bp /opt/anaconda \
    && rm /root/miniconda.sh
ENV PATH /opt/anaconda/bin:$PATH

COPY conda-env.yaml /root/
RUN conda install anaconda \
    && conda env update -n root -f /root/conda-env.yaml \
    && conda clean -tipsy \
    && rm -fr \
        /root/conda-env.yaml \
        /root/.cache

WORKDIR /app
RUN chmod a+wx /app
ENV HOME /app
CMD [ "python" ]
