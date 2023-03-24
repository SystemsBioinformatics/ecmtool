#!/usr/bin/env bash

docker build -t ecmtool -f ./docker/Dockerfile .
docker run ecmtool python3 tests/test_conversions.py
