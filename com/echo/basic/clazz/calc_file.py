

def scan_file():
    status = {}
    liantong = open("D:\data\中国联通.csv", encoding="utf-8")

    while line := liantong.readline():
        strs = line.split("|")
        str = strs[0]
        status[str] = status.get(str, 0) + 1

    print(status)



def split_file():
    liantong = open("D:\data\中国联通.csv", encoding="utf-8")
    result = open(r"D:\data\result\liantong.csv", "w+", encoding="gbk")
    result.write("dataused,userid0,spcode1,pay2,province3, cardtype4, industry5, devicetype6,customerstatus7, voice8, message9, data10, voicelimtype11, messagelimtype12,datalimtype13, bindstgate14, bindtype15, fixuretype16, risklevel17, dataused18\n")
    count = 1
    size = 0

    while line := liantong.readline():
        # userid0,spcode1,pay2,province3, cardtype4, industry5, devicetype6,
        # customerstatus7, voice8, message9, data10, voicelimtype11, messagelimtype12,
        # datalimtype13, bindstgate14, bindtype15, fixuretype16, risklevel17, dataused18
        # 89860620160039957041 | 3 | 2 | HE | 1 | 1 | 12 | 已激活 | 0 | 0 | 1 | | | 4 | 1 | 1 | 2 | 3 | 616762.0
        # size = size + 1
        # print(f"\r {size}", end="")

        if count % 1000000 == 0:
            result.flush()
            result.close()
            result = open(fr"D:\data\result\liantong{count}.csv", "w+", encoding="gbk")
            result.write("dataused,userid0,spcode1,pay2,province3, cardtype4, industry5, devicetype6,customerstatus7, voice8, message9, data10, voicelimtype11, messagelimtype12,datalimtype13, bindstgate14, bindtype15, fixuretype16, risklevel17, dataused18\n")

        strs = line.split("|")
        if strs[4] != '':
            line = line.replace("|", ",")
            result.write(f"{eval(strs[18])},{line}")
            result.flush()
            count = count + 1

    result.flush()
    result.close()
    liantong.close()

if __name__ == '__main__':
    split_file()