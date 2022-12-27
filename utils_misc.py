import hashlib
import os
import platform
import shutil
import sys
import time
from enum import Enum
from typing import Any, Union, Optional, Type, Callable

import math
import numpy as np

from logger import error, exception, warn, info
from utils_random import randInt, random

APP_RUN_ID = None


def getPythonVersion(getTuple: bool = False) -> Union[str, tuple]:
    version = sys.version_info
    version_tuple = (version.major, version.minor, version.micro)
    if getTuple:
        return version.major, version.minor, version.micro
    else:
        return '.'.join([str(el) for el in version_tuple])


def getPythonExecName() -> str:
    version = getPythonVersion(getTuple=True)
    full_name = f'python{version[0]}.{version[1]}'
    short_name = f'python{version[0]}'
    default_name = 'python'
    if shutil.which(full_name) is not None:
        return full_name
    if shutil.which(short_name) is not None:
        return short_name
    return default_name


def getOsName() -> str:
    os_name = sys.platform
    if 'linux' in os_name:
        return 'linux'
    elif 'darwin' in os_name:
        return 'mac'
    elif 'win' in os_name:
        return 'windows'
    else:
        return os_name


def isAppleSilicon() -> bool:
    if getOsName() != 'mac':
        return False
    return platform.processor() == 'arm'


def getRunId() -> int:
    global APP_RUN_ID
    if APP_RUN_ID is None:
        APP_RUN_ID = randInt(666666, force_rng=True)
    return APP_RUN_ID


def getRunIdStr() -> str:
    return f'run_id-{getRunId():06d}'


def getFibonacciSeq(size: int) -> list[int]:
    if size <= 1:
        return [0]
    fibonacci = [0, 1]
    for i in range(2, size, 1):
        fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
    return fibonacci


def hashStr(string: str, low_resolution: bool = False) -> str:
    to_hash = string.encode('utf-8')
    if low_resolution:
        hash_object = hashlib.md5(to_hash)
    else:
        hash_object = hashlib.sha256(to_hash)
    return hash_object.hexdigest()


def listToChunks(array: list[Any], n_chunks: Optional[int] = None, chunk_sz: Optional[int] = None,
                 filter_empty: bool = False) -> list[list]:
    if chunk_sz is not None:
        c_n = math.ceil(len(array) / chunk_sz)
        c_size = chunk_sz
    elif n_chunks is not None:
        c_size = math.ceil(len(array) / n_chunks)
        c_n = n_chunks
    else:
        raise AttributeError('You must provide either chunk_sz or n_chunks')
    chunks = [array[i * c_size:i * c_size + c_size] for i in range(c_n)]
    if filter_empty:
        chunks = list(filter(lambda x: len(x) > 0, chunks))
    return chunks


_numpy_list_numeric_types_union = Union[np.ndarray, list, np.float64, np.float32, np.int64, np.int32]


def numpyToListRecursive(element: _numpy_list_numeric_types_union) -> _numpy_list_numeric_types_union:
    if type(element) is np.ndarray:
        return numpyToListRecursive(element.tolist())
    elif type(element) is list:
        if len(element) == 0:
            return element
        elif type(element[0]) in (np.float64, np.float32):
            return list(map(float, element))
        elif type(element[0]) in (np.int64, np.int32):
            return list(map(int, element))
        elif type(element[0]) in (int, float):
            return element
        else:
            raise Exception(f'Unhandled el type {type(element[0])}')
    else:
        raise Exception(f'Unhandled type {type(element)}')


def _add_two_way_dict(hash_table: dict, k: Any, v: Any) -> None:
    hash_table[k] = v
    hash_table[v] = k


def _add_three_way_dict(hash_table: dict, k1: Any, k2: Any, v: Any) -> None:
    _add_two_way_dict(hash_table, k1, v)
    _add_two_way_dict(hash_table, k2, v)


def _add_four_way_dict(hash_table: dict, k1: Any, k2: Any, k3: Any, v: Any) -> None:
    _add_two_way_dict(hash_table, k1, v)
    _add_two_way_dict(hash_table, k2, v)
    _add_two_way_dict(hash_table, k3, v)


def safeLog(x: Any) -> Any:
    if type(x) not in getNumericTypes():
        return x
    return math.log(x)


def safeSum(arr: list[Any], get_count: bool = False) -> Union[tuple, Optional[float]]:
    numerics = 0
    summation = 0
    for x in arr:
        if type(x) in getNumericTypes():
            summation += x
            numerics += 1
    if numerics > 0:
        if get_count:
            return summation, numerics
        else:
            return summation
    else:
        if get_count:
            return None, numerics
        else:
            return summation


def safeMean(array: list[float]) -> Optional[float]:
    summation, count = safeSum(array, get_count=True)
    if summation is not None:
        return summation / count
    else:
        return summation


def weightedAverage(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights):
        raise AttributeError('Error, mismatching size between values and weights!')
    w_sum = sum(weights)
    v_sum = 0
    for v, w in zip(values, weights):
        v_sum += v * w
    return v_sum / w_sum


def getEnumRange(the_enum: Type[Enum], list_output: bool = False) -> Union[dict, list[int, int]]:
    all_enums = list(the_enum._member_map_.items())
    min_value = the_enum(all_enums[0][1]).value
    max_value = the_enum(all_enums[-1][1]).value
    if list_output:
        return [min_value, max_value]
    else:
        return dict(min_value=min_value, max_value=max_value)


def getNumericTypes() -> tuple:
    return int, float, np.int32, np.int64, np.float32, np.float64


def isNumericString(string: str) -> bool:
    return string.replace('.', '0').isnumeric()


def anythingToBool(anything: Any) -> bool:
    positive_check = anything in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade', True, 1)
    if positive_check:
        return True
    numeric_positive_check = (type(anything) in getNumericTypes() and anything > 0) or \
                             (type(anything) is str and isNumericString(anything) and float(anything) > 0)
    return numeric_positive_check


def getCpuCount() -> int:
    return os.cpu_count()


def binarySearch(lis: list, el: Any) -> Any:  # list must be sorted
    low = 0
    high = len(lis) - 1
    ret = None
    while low <= high:
        mid = (low + high) // 2
        if el < lis[mid]:
            high = mid - 1
        elif el > lis[mid]:
            low = mid + 1
        else:
            ret = mid
            break
    return ret


def base64ToBase65(base64: str, char_65: str = '*') -> str:
    base65 = ''
    mark_char = '&'
    last_char = mark_char
    min_length = 4
    count = 1
    for c in base64 + last_char:
        if last_char == mark_char:
            last_char = c
            count = 1
        else:
            if last_char != c:
                if count > min_length:
                    base65 += f'{last_char}{char_65}{count}{char_65}'
                else:
                    base65 += last_char * count
                count = 0
                last_char = c
            count += 1
    return base65


def base65ToBase64(base65: str, char_65: str = '*') -> str:
    if char_65 not in base65:
        return base65
    base64 = ''
    mark_char = '&'
    last_char = mark_char
    count = None
    for c in base65:
        if c == char_65:
            if count is None:
                count = -1
            else:
                base64 += last_char * (count - 1)
                count = None
        if count is None:
            if c != char_65:
                base64 += c
                last_char = c
        else:
            if c != char_65:
                if count == -1:
                    count = int(c)
                else:
                    count *= 10
                    count += int(c)
    return base64


def mergeDicts(high_priority: dict, low_priority: dict) -> dict:
    return {**low_priority, **high_priority}


def exponentialBackoff(cur_retry: int = 0, max_attempts: int = 5, backoff_in_secs: float = 0.1):
    # cur_retry starts with zero
    if cur_retry < max_attempts:
        exponential_delay = (2 ** cur_retry) * backoff_in_secs
        random_delay = random()
        time.sleep(exponential_delay + random_delay)
    else:
        raise TimeoutError(f'Too many retries (`{max_attempts}`)')
    cur_retry += 1
    return cur_retry


def exceptionExpRetry(name: str, function: Callable, args: list, kwargs: dict, max_attempts: int,
                      backoff_s: float = 0.1, raise_it: bool = True) -> Any:
    t = 0
    while True:
        try:
            res = function(*args, **kwargs)
            if t > 0:
                info(f'Succeeded to run `{name}` on attempt {t + 1}!')
            return res
        except Exception as e:
            try:
                t = exponentialBackoff(t, max_attempts, backoff_s)
                warn(f'Failed to run `{name}`, attempt {t} of {max_attempts}!')
            except TimeoutError:
                error(f'Failed to run `{name}` after {max_attempts} attempts!')
                if raise_it:
                    raise e
                else:
                    exception(e, False)
