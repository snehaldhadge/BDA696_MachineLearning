FROM ubuntu

RUN mkdir /scripts
COPY ./assign5_bash.sh /scripts/assign5_bash.sh
COPY ./baseball.sql /scripts/baseball.sql
COPY ./assign5_sql.sql /scripts/assign5_sql.sql
RUN chown 1000:1000 /scripts

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     mysql-client \
     wget \
  && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip3 install --compile --no-cache-dir -r requirements.txt

# RUN wget teaching.mrsharky.com/data/baseball.sql.tar.gz
# RUN tar -xvzf baseball.sql.tar.gz

RUN chmod +x /scripts/assign5_bash.sh
CMD ./scripts/assign5_bash.sh



