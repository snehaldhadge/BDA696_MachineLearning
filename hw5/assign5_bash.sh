#!/bin/sh

sleep 20

if ! mysql -h my-db -uroot -e 'use baseball'; then
  echo "Baseball DOES NOT exists"
    mysql -h my-db -u root -e "CREATE DATABASE IF NOT EXISTS baseball"
    mysql -h my-db -u root baseball < /scripts/baseball.sql
else
  echo "Baseball DOES exists"
fi

echo "Calling assign5_sql"
mysql -h my-db -u root  baseball < /scripts/assign5_sql.sql

mysql -h my-db -u root baseball -e '
  SELECT * FROM rolling_average;' > /results/output_alldata_assign5.txt

mysql -h my-db -u root baseball -e '
  SELECT * FROM rolling_average where game_id=12560;' > /results/output_assign5.txt
