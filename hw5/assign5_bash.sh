#!/bin/sh

RESULT=`mysqlshow -h my_db  -u root baseball| grep -v Wildcard | grep -o baseball`
if [ "$RESULT" == "baseball" ]; then
    echo "Database exists"
else	
    mysql -h my_db  -u root  -e "CREATE DATABASE IF NOT EXISTS baseball"
    mysql -h my_db -u root  baseball < /scripts/baseball.sql
fi
echo "Calling assign5_sql"
mysql -h my_db -u root  baseball < /scripts/assign5_sql.sql
