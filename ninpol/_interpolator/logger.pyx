import json
import re
import os
import sys

import numpy as np

from datetime import datetime
from _ctypes import PyObj_FromPtr

from ..utils import common

cdef class Logger:
    def __cinit__(self, str log_name, str directory=""):
        self.last_index = 0
        cdef str suffix = "-" + datetime.now().strftime('%y%m%d')

        self.filename = log_name + suffix + "_" + str(self.last_index) + ".log"
        self.json_filename = log_name + suffix + "_" + str(self.last_index) + ".json"
        self.directory = directory

        if self.directory == "":
            # Create .ninpollog directory at the terminal's current directory
            self.directory = os.getcwd()
            self.directory = os.path.join(self.directory, ".ninpollog")

        self.filename = os.path.join(self.directory, self.filename)
        self.json_filename = os.path.join(self.directory, self.json_filename)

        self.data = {}
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        else:
            # If there's a log file with the same name, set the last_index to the next number
            while os.path.exists(self.filename) or os.path.exists(self.json_filename):
                self.last_index += 1
                self.filename = log_name + suffix + "_" + str(self.last_index) + ".log"
                self.json_filename = log_name + suffix + "_" + str(self.last_index) + ".json"
                self.filename = os.path.join(self.directory, self.filename)
                self.json_filename = os.path.join(self.directory, self.json_filename)
    

    def log(self, str message, str type):
        with open(self.filename, 'a') as f:
            f.write(f"[{type:<5}] ({datetime.now().strftime('%H:%M:%S'):<8}) {message}\n")

    def np_to_list(self, data):
        # Convert all keys to string
        sdata = {str(key): value for key, value in data.items()}
        for key, value in sdata.items():
            if isinstance(value, np.ndarray):
                sdata[key] = common.arr_to_dict(value)
            elif isinstance(value, dict):
                sdata[key] = self.np_to_list(value)
        return sdata

    def json(self, str member_name, dict data):
        data = self.np_to_list(data)
        new_data = {"timestamp": datetime.now().strftime('%H:%M:%S'), "data": data}
        self.data[member_name] = new_data
        if self.json_filename:
            with open(self.json_filename, 'a') as f:
                self.pretty_json(self.data, f)
    
    def pretty_json(self, dict data, f):
        s = json.dumps(data, cls=CustomJSONEncoder, indent=2)
       
        f.seek(0)
        f.truncate()
        f.write(s + '\n')

class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        self._indent = kwargs.pop('indent', None)
        super().__init__(*args, **kwargs)
    
    def encode(self, o, level=0):
        if isinstance(o, list):
            if all(not isinstance(el, (list, dict)) for el in o):
                return f'[{", ".join(map(str, o))}]'
            else:
                nested_items = [self.encode(el, level + 1) for el in o]
                indented_items = [(self._indent * (level + 1)) * ' ' + item for item in nested_items]
                return '[\n' + ',\n'.join(indented_items) + '\n' + (self._indent * level) * ' ' + ']'
        elif isinstance(o, dict):
            items = [
                f'{(self._indent * (level + 1)) * " "}{json.dumps(k)}: {self.encode(v, level + 1)}'
                for k, v in o.items()
            ]
            return '{\n' + ',\n'.join(items) + '\n' + (self._indent * level) * ' ' + '}'
        return json.dumps(o)



