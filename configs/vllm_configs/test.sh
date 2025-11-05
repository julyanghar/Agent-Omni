#!/bin/bash

MODEL=$1
COMMAND=$(yq -r '.vllm_commands["'"$MODEL"'"]' < vllm_config.yaml)


echo "Running: $COMMAND"
eval "$COMMAND"

