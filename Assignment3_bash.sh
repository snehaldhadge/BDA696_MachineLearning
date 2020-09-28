#!/bin/bash

# Generating the requirement.dev file
Echo "Installing all the packages"
pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

pip3 install -r requirements.dev.txt

pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

echo "Calling the Python Code"
# Running the Python file
python3 ./Assignment3_Snehal.py
