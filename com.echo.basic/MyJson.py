import json

if __name__ == '__main__':

    # 转换json
    phones = {"echo": "15210012571", "liudr": "13253329379"}
    str = json.dumps(phones)

    print(str)

    # 解析json
    jsonString = '{"code":200,"msg":"success","data":[]}'
    obj = json.loads(jsonString)

    print(obj['code'])







