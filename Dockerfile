FROM bempp/notebook

#COPY /etc/resolv.conf /etc/resolv.conf

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get install -y python3-tk && \
    chown -R root:root ~/.cache && \
    pip install -U setuptools pip && \
    mkdir $HOME/.pip && \
    pip install -U scipy cython --ignore-installed && \
    # printf "[global]\nextra-index-url = https://www.nag.com/downloads/py/naginterfaces_nag\n" > $HOME/.pip/pip.conf