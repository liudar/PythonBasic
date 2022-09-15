import pandas as pd
from pandasql import sqldf
import subprocess

'''
sql 很方便， 但是写的内容有点长
'''
def sql():
    data = pd.read_csv("./data.csv",delimiter='\t')
    # 声明为全局变量
    # global data
    # 一次行声明好全局变量，然后用这个返回值调用sql
    # pysqldf = lambda q: sqldf(q,globals())

    # result = sqldf("select * from data order by one desc limit 5")
    result = sqldf("select * from data group by one")
    print(result)

def py():
    data = pd.read_csv(r"./data.csv",delimiter='\t')

    # 过滤
    devicetype11 = data[data['devicetype'] == '11']
    print(devicetype11)

    count = data.groupby('devicetype')['sum'].transform('count')
    print(count)

    # 按照devicetype汇总后将所有字段求和
    data.groupby('devicetype').sum()
sql = """
        aaa
        bbb
        bccc
    """
def exec(date):
    print(sql)

exec(123)
if __name__ == '__main__':
    # sql()
    # data = pd.read_csv(r"C:\Users\echo\Desktop\基础字段\运营商-省份.csv", delimiter=',', encoding='gbk')
    # # result = sqldf("select * from data limit 10;")
    # result = sqldf("select * from data group by province_value")
    # print(result)

    subprocess.call("ls ./",shell=True)

    print("xxx && hive -e \"sql\"")

    
