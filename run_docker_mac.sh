#!/bin/sh

docker build -t ecmtool ./docker/
docker run -ti ecmtool
