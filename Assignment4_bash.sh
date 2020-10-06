#!/bin/bash
pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

pip3 install -r requirements.dev.txt

pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

read -p "Enter CSV Name with FullPath (Leave blank if using existing csv):" fname
read -p "Enter the response variable name" target
if [[ $target == '' ]] && [[ $fname != '' ]];then
  read -p "Please enter a response/target variable:" target
  if [ -z $target ];then
     '$fname' = ''
  fi
fi

echo $fname

echo "Calling Python Code"
python3 Assignment_4.py $fname $target
