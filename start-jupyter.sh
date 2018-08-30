#!/usr/bin/env bash
dir=$(dirname $0)

pipenv run jupyter lab \
    --NotebookApp.notebook_dir="$dir" \
    --NotebookApp.open_browser=False \
    --NotebookApp.ip=*
