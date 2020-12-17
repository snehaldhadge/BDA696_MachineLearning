FROM ubuntu

RUN mkdir /scripts
COPY ./Bash_Script_SQL.sh /scripts/Bash_Script_SQL.sh
COPY ./dbdata/IPL.sql /scripts/IPL.sql
#COPY ./dbdata/small_ipl.sql /scripts/small_ipl.sql
COPY ./Brute_Force_Plots.py /scripts/Brute_Force_Plots.py
COPY ./table_gen_sql.sql /scripts/table_gen_sql.sql
COPY ./IPL_project.py /scripts/IPL_project.py
#COPY ./small.sql /scripts/small.sql
#COPY ./try_sql_conn.py /scripts/try_sql_conn.py
COPY ./Predictor_Plots_Ranking.py /scripts/Predictor_Plots_Ranking.py
RUN chown 1000:1000 /scripts

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     libmysqlclient-dev \
     mysql-client \
     python3 \
     python3-pip \
     python3-dev \
     python3-pymysql \
     wget \
  && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

RUN chmod +x /scripts/Bash_Script_SQL.sh
#RUN chmod +x /scripts/try_sql_conn.py
CMD ./scripts/Bash_Script_SQL.sh
#CMD python3 ./scripts/try_sql_conn.py
# Calling the Python script to generate Model
#CMD python3 ./scripts/IPL_project.py


