FROM alpine:3.11

RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories
RUN apk update && apk add git openssh build-base bash openjdk14 openmpi-dev gmp-dev python3 python3-dev cython py3-numpy py3-numpy-dev py3-scipy py3-psutil py3-scikit-learn
RUN git clone https://github.com/tjclement/ecmtool.git
RUN cd ecmtool && pip3 install -r requirements.txt && python3 setup.py install && cp /usr/lib/python3.*/site-packages/ecmtool*.egg/_bglu_dense.cpython-38-x86_64-linux-gnu.so ecmtool/
RUN cd .. && wget http://cgm.cs.mcgill.ca/~avis/C/lrslib/archive/lrslib-070a.tar.gz && tar -xzf lrslib-070a.tar.gz && cd lrslib-070 && make && make install && cd .. && rm -rf lrslib*

WORKDIR /ecmtool

CMD ["/bin/ash"]
