#!/bin/sh

docker build -t ecmtool -f ./docker/Dockerfile .
docker run -ti ecmtool
