from __future__ import annotations
import random
import time
from typing import Callable, TypeVar, Iterable

T = TypeVar("T")

def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_sleep_s: float = 0.25,
    max_sleep_s: float = 2.0,
    jitter: float = 0.25,
    retry_on: Iterable[type[Exception]] = (Exception,),
) -> T:
    
    last_err: Exception | None = None

    for i in range(1, attempts + 1):
        try:
            return fn()
        except tuple(retry_on) as e:
            last_err = e
            if i == attempts:
                break

            # exponential backoff with jitter
            sleep = min(max_sleep_s, base_sleep_s * (2 ** (i - 1)))
            sleep = sleep * (1.0 + random.uniform(-jitter, jitter))
            time.sleep(max(0.0, sleep))

    assert last_err is not None
    raise last_err
