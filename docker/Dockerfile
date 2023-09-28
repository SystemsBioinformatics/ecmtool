FROM python:3.10-slim

RUN apt update && apt install -y git openssh-client build-essential bash openjdk-17-jre libopenmpi-dev libgmp-dev wget lrslib

COPY ecmtool /ecmtool/ecmtool/
COPY models /ecmtool/models/
COPY tests /ecmtool/tests/
COPY tmp/.gitignore /ecmtool/tmp/
COPY main.py setup.py cython_setup.py requirements.txt /ecmtool/
WORKDIR /ecmtool

RUN pip3 install -r requirements.txt && pip3 install mpi4py==3.1.4
RUN python3 cython_setup.py build_ext --inplace && mv *.so ecmtool
RUN python3 setup.py install

# Make sure we don't have to run as root
RUN useradd -ms /bin/bash user
RUN chown -R user:user /ecmtool
USER user

CMD ["/bin/bash"]
