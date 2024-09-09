import numpy as np

def arr_to_dict(arr):
    dict = {}
    dict['shape'] = arr.shape
    dict['dtype'] = str(arr.dtype)
    dict['data']  = arr.tolist()
    return dict