#!/bin/bash
set -e
wget https://downloads.apache.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz
apt install default-jre
tar -zxvf spark-3.0.1-bin-hadoop2.7.tgz
mkdir -p /opt/apache-spark/
mkdir -p ./data
mv ./spark-3.0.1-bin-hadoop2.7 /opt/apache-spark/
pip install -r requirements.txt
rm -rf spark-3.0.1-bin-hadoop2.7.tgz