import random as rd
import time
import uuid
from typing import Union, Iterable, Any, Optional

from numpy.random import Generator, MT19937


class RngHolder(object):
    RNG = None

    @staticmethod
    def getRng(force_new: bool = False) -> Generator:
        if force_new or RngHolder.RNG is None:
            seed = int(time.perf_counter() * rd.random() + time.time())
            if RngHolder.RNG is not None:
                seed += RngHolder.RNG.integers(int(time.time()))
            RngHolder.RNG = Generator(MT19937(seed))
        return RngHolder.RNG


def random():
    return RngHolder.getRng().random()


def randInt(high_inc: int = 0, low: int = 0, size: int = 1, high_exc: Optional[int] = None,
            force_rng: bool = False) -> Union[int, list[int]]:
    if high_exc is not None:
        high = high_exc
    else:
        high = high_inc + 1
    out = RngHolder.getRng(force_new=force_rng).integers(low=low, high=high, size=size)
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
    return RngHolder.getRng().choice(array, size=amount)


def randShuffle(array: Iterable[Any], axis: int = 0) -> None:
    RngHolder.getRng().shuffle(array, axis=axis)


def randomUUID() -> str:
    return uuid.uuid4().hex
