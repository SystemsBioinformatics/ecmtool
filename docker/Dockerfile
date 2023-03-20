FROM python:3.10-slim

RUN apt update && apt install -y git openssh-client build-essential bash openjdk-17-jre libopenmpi-dev libgmp-dev wget
COPY / /ecmtool
RUN cd ecmtool && pip3 install -r requirements.txt && python3 cython_setup.py build_ext --inplace && mv *.so ecmtool && python3 setup.py install
RUN cd .. && wget http://cgm.cs.mcgill.ca/~avis/C/lrslib/archive/lrslib-071a.tar.gz && tar -xzf lrslib-071a.tar.gz && cd lrslib-071a && make && make mplrs && make install && ln -s `pwd`/mplrs /usr/local/bin/mplrs && ln -s `pwd`/redund /usr/local/bin/redund

# Make sure we don't have to run as root
RUN useradd -ms /bin/bash user
RUN chown -R user:user /ecmtool /lrslib*
USER user
WORKDIR /ecmtool

CMD ["/bin/bash"]
