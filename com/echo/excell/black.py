import xlrd
import xlwt
import pandas as pd
import numpy as py

def exec(data):
    return 0

if __name__ == '__main__':
    file = xlrd.open_workbook("./black.xlsx")
    print(file.sheets())
    sheets = file.get_sheets()

    for i in len(sheets):
        print(i)

