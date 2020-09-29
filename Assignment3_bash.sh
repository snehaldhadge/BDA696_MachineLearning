#!/bin/bash

wget https://downloads.mariadb.com/Connectors/java/connector-java-2.4.3/mariadb-java-client-2.4.3.jar
mv  mariadb-java-client-2.4.3.jar .venv/lib/python3.8/site-packages/pyspark/jars/
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
