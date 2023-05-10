#!/usr/bin/env bash

docker build -t ecmtool -f ./docker/Dockerfile .
docker run ecmtool python3 -m pytest tests/test_conversions.py
