import functools
from pylab import mpl


def enable_chinese(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams["axes.unicode_minus"] = False
        return func(*args, **kwargs)
    return wrapper
