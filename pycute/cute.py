#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Computes the SNML code for a sequence. We use log-sum-exp trick  to fix the math overflow/underflow error.
https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes
"""
from typing import List, Tuple

from .utils import lg, exp

__all__ = ['regret', 'bernoulli', 'cbernoulli', 'cute']


def regret(t1: int, t: int, x: int) -> float:
    t0 = t - 1 - t1
    ll1 = (t1 + 1) * lg(t1 + 1) + t0 * lg(t0)
    ll0 = t1 * lg(t1) + (t0 + 1) * lg(t0 + 1)
    max_ll = max(ll1, ll0)
    lg_numer = ll1 if x == 1 else ll0
    lg_denom = max_ll + lg(exp(ll1 - max_ll) + exp(ll0 - max_ll))
    return lg_denom - lg_numer


def bernoulli(X: List[int]) -> float:
    t1 = 0
    res = 0.0
    for t, x in enumerate(X, 1):
        if t == 1:
            t1 += int(x == 1)
            continue
        res += regret(t1, t, x)
        t1 += int(x == 1)
    return res


def cbernoulli(X: List[int], Y: List[int]) -> float:
    res = 0.0
    t_x, t_y, t_max = 0, 0, 0
    for t, x in enumerate(X, 1):
        y = Y[t - 1]
        if t == 1:
            t_x += int(x == 1)
            t_y += int(y == 1)
            t_max = t_x or t_y
            continue

        t1 = min(t_x, t_y) if x == 0 else t_max
        res += regret(t1, t, x)

        t_x += int(x == 1)
        t_y += int(y == 1)
        t_max += int(x == 1) or int(y == 1)
    return res


def cute(X: List[int], Y: List[int]) -> Tuple[float, float]:
    delta_XtoY = bernoulli(Y) - cbernoulli(Y, X)
    delta_YtoX = bernoulli(X) - cbernoulli(X, Y)
    return delta_XtoY, delta_YtoX
