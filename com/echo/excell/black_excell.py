import pandas as pd

if __name__ == '__main__':
    lines = open("./temp.csv").readlines()
    result = open("./temp2.csv", "w+")
    maps = {}
    operatorCodes = {
        "1": "中国电信",
        "2": "中国移动",
        "3": "中国联通",
    }
    provinceCodes = {
        "AH": "安徽",
        "BJ": "北京",
        "CQ": "重庆",
        "FJ": "福建",
        "GD": "广东",
        "GS": "甘肃",
        "GX": "广西壮族自治区",
        "GZ": "贵州",
        "HA": "河南",
        "HB": "湖北",
        "HE": "河北",
        "HN": "湖南",
        "JL": "吉林",
        "JS": "江苏",
        "JX": "江西",
        "LN": "辽宁",
        "NX": "宁夏回族自治区",
        "QH": "青海",
        "SC": "四川",
        "SD": "山东",
        "SH": "上海",
        "SN": "陕西",
        "SX": "山西",
        "TJ": "天津",
        "XJ": "新疆维吾尔自治区",
        "XZ": "西藏自治区",
        "YN": "云南",
        "ZJ": "浙江",
        "AM": "澳门",
        "XG": "香港",
        "HL": "黑龙江",
        "NM": "内蒙古自治区",
        "HI": "海南",
    }
    provinces = [
        "北京",
        "天津",
        "河北",
        "山西",
        "内蒙古自治区",
        "辽宁",
        "吉林",
        "黑龙江",
        "上海",
        "江苏",
        "浙江",
        "安徽",
        "福建",
        "江西",
        "山东",
        "河南",
        "湖北",
        "湖南",
        "广东",
        "广西壮族自治区",
        "海南",
        "重庆",
        "四川",
        "贵州",
        "云南",
        "西藏自治区",
        "陕西",
        "甘肃",
        "青海",
        "宁夏回族自治区",
        "新疆维吾尔自治区"
    ]
    for line in lines:
        words = line.replace("\n", "").split("\t")

        if (int(words[2]) != 0):
            cp1 = int(words[3]) / int(words[2]) * 100
        else:
            cp1 = 0

        if (int(words[4]) != 0):
            cp2 = int(words[5]) / int(words[4]) * 100
        else:
            cp2 = 0
        key = provinceCodes.get(words[1],"") + "," + operatorCodes.get(words[0]) + ","
        maps[key] = f"{words[2]},{words[3]},{'%.2f'%cp1}%,{words[4]},{words[5]},{'%.2f'%cp2}%"

    for province in provinces:
        for i in range(3):
            key = province + "," + operatorCodes.get(str(i + 1)) + ","
            result.write(key + maps.get(key, "") + "\n")
            result.flush()
