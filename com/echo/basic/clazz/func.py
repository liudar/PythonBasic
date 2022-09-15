from functools import wraps

# 跟踪方法
def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 调用方法
        result = func(*args, **kwargs)
        print(f'{func.__name__}({args!r}, {kwargs!r}) '
              f'-> {result!r}')
        return result
    return wrapper

@trace
def aaa():
    print(123)
    return 123


if __name__ == '__main__':
    aaa()