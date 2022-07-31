import joblib

def add(x, y):
    return x + y

if __name__ == '__main__':
    print(add("2", "1"))
    joblib.dump("1", "aaa")
    b = joblib.load("aaa")
    print(b)

