import os
import datetime
import subprocess

source_local = "source /opt/hadoopclient/bigdata_env && kinit -kt /opt/hadoopclient/user/user.keytab hive_hdfs && "

sql = "insert into table xxx partition(cp='2022090300') select XXX"

def exec(date):
    global sql
    # 每一个替换都很重要
    sql = sql.replace("\n"," ").replace("2022090300",date).replace("`","\\`").replace("\t"," ").replace("\"","\\\"")
    result = subprocess.call(f"{source_local} beeline -e \"{sql}\"", shell=True)
    print(result)

exec("2022080100")