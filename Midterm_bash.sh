#!/bin/bash
pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

pip3 install -r requirements.dev.txt

pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

echo "Choose your option:"
echo "1)Choose Sklearn Dataset"
echo "2)Read from a CSV"
echo "3)Read from default CSV (Auto_Mpg)"

read n

case $n in
  1) echo "Select from following dataset:"
     echo "1) load_boston"
     echo "2) load_diabetes"
     echo "3) load_breast_cancer"
     read d
     fname=$n
     target=$d
     ;;
  2) read -p "Enter CSV Name with FullPath (Leave blank if using existing csv):" fname
     read -p "Enter the response variable name" target
     if [[ $target == '' ]] && [[ $fname != '' ]];then
        read -p "Please enter a response/target variable:" target
      if [ -z $target ];then
            '$fname' = ''
      fi
     fi;;
  3) '$fname' = ''
     '$target' = '';;
esac

echo "Calling Python Code"
python3 MidTerm.py $fname $target


