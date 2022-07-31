
import calendar
import sqlite3
import requests

def request(url):
    resp = requests.get(url.split(" ")[1])
    print(resp.text)


def date(msg):
    ss = msg.split(" ")
    if ss[0] == 'month':
        year = 2022
        month = 1
        ss2 = ss[1].split("-")
        if len(ss2) == 1:
            month = int(ss2[0])
        else:
            year = int(ss2[0])
            month = int(ss2[1])
        print(calendar.month(year, month))

# sql
# create table teacher(name varchar(20))
# insert into teacher values("echo")
# select * from teacher
def sqlite():
    con = sqlite3.connect(':memory:')

    while True:
        sql = input("sql:: ")

        if sql == 'exit':
            break

        cursor = con.execute(sql)
        print(cursor.fetchall())

    # con.execute('create table test(a varchar(20))')
    # con.commit
    #
    # con.execute("insert into test values('liudr')")
    # con.commit
    #
    # cursor = con.execute('select * from test')
    # print(cursor.fetchall())
    # print(cursor.description)



if __name__ == '__main__':
    print("欢迎来到我的客户端")
    while True:
        text = input(">>> ")

        if text.startswith('month'):
            date(text)
        elif text == 'sql':
            sqlite()
        elif text.startswith('request'):
            request(text)
        else:
            print(text)