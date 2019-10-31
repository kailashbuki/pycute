#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Helper methods.
"""
from math import log
from typing import Union


def lg(x: Union[int, float]) -> float:
    """Computes the logarithm of a number base 2. We enforce log0=0.
    """
    res = 0.0
    try:
        res = log(x, 2)
    except ValueError:
        pass
    return res


def exp(x: Union[int, float]) -> float:
    return 2 ** x
