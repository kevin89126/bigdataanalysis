#!/bin/bash
set -e
wget https://downloads.apache.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz
apt install default-jre cron
tar -zxvf spark-3.0.1-bin-hadoop2.7.tgz
mkdir -p /opt/apache-spark/
mkdir -p ./data
mv ./spark-3.0.1-bin-hadoop2.7 /opt/apache-spark/
pip install -r requirements.txt
chmod +x /root/bigdataanalysis/LSTM_FINAL.py
echo "0 10 * * 6 root /root/bigdataanalysis/LSTM_FINAL.py 2>&1 | tee -a /nfs/Workspace/LSTM_FINAL.log" >> /etc/crontab
service cron start
rm -rf spark-3.0.1-bin-hadoop2.7.tgz
