import base64
import codecs
import json
from enum import Enum
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


class NumpyJsonEncoder(json.JSONEncoder):

    @staticmethod
    def defaultEnumNumericBool(enc, obj: Any, next_enc: Callable) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if type(obj) in (np.int32, np.int64):
            return int(obj)
        if type(obj) in (np.float32, np.float64):
            return float(obj)
        if type(obj) is np.bool_:
            return bool(obj)
        return next_enc(enc, obj)

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return NumpyJsonEncoder.defaultEnumNumericBool(self, obj, json.JSONEncoder.default)

    class Compact(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                data_b64 = base64.b64encode(obj.data).decode()
                return dict(__ndarray__=data_b64,
                            dtype=str(obj.dtype),
                            shape=obj.shape)
            return NumpyJsonEncoder.defaultEnumNumericBool(self, obj, json.JSONEncoder.default)

        @staticmethod
        def dec(dct: Any) -> Any:
            if isinstance(dct, dict) and '__ndarray__' in dct:
                data = base64.b64decode(dct['__ndarray__'].encode())
                return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
            return dct


def saveObj(obj: Any, filepath: str) -> None:
    joblib.dump(obj, filepath)


def loadObj(filepath: str) -> Any:
    return joblib.load(filepath)


def saveDataframe(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, encoding='utf-8', index=False)


def loadDataframe(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def loadDataframeColumns(filepath: str, separator: str = ',') -> list[str]:
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line != '':
                return line.split(separator)
    return []


def saveJson(json_dict: dict, filepath: str, sort_keys: bool = True, indent: bool = True) -> None:
    with codecs.open(filepath, 'w', encoding='utf-8') as file:
        kwargs = {}
        if indent:
            kwargs['indent'] = 4
        json.dump(json_dict, file, separators=(',', ':'), sort_keys=sort_keys, cls=NumpyJsonEncoder, **kwargs)


def loadJson(filepath: str, object_hook: Callable = None) -> Any:
    with codecs.open(filepath, 'r', encoding='utf-8') as file:
        json_dict = json.loads(file.read(), object_hook=object_hook)
    return json_dict
