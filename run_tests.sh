#!/usr/bin/env bash

docker build -t ecmtool -f ./docker/Dockerfile .
docker run ecmtool python3 main.py --model_path models/e_coli_core.xml --auto_direction true --direct true --polco true