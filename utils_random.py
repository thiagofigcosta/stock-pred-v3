import random as rd
import time
import uuid
from typing import Union, Iterable, Any, Optional

from numpy.random import Generator, MT19937

RNG = Generator(MT19937(int(time.perf_counter() * rd.random())))


def random():
    return RNG.random()


def randInt(high_inc: int = 0, low: int = 0, size: int = 1, high_exc: Optional[int] = None) -> Union[int, list[int]]:
    if high_exc is not None:
        high = high_exc
    else:
        high = high_inc + 1
    out = RNG.integers(low=low, high=high, size=size)
    if size == 1:
        out = out[0]
    return out


def randomFloat(high: int, low: int = 0, size: int = 1) -> Union[float, list[float]]:
    out = []
    for _ in range(size):
        out.append(low + (random() * (high - low)))
    if size == 1:
        out = out[0]
    return out


def randChoice(array: Iterable[Any], amount: int = 1) -> Any:
    if amount <= 1:
        amount = None
    if type(array) is set:
        array = list(array)
    return RNG.choice(array, size=amount)


def randShuffle(array: Iterable[Any], axis: int = 0) -> None:
    RNG.shuffle(array, axis=axis)


def randomUUID() -> str:
    return uuid.uuid4().hex
