#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Computes the transfer entropy from Y to X.
"""
from collections import Counter, defaultdict
from typing import DefaultDict, List, Tuple

import numpy as np  # type: ignore

from .utils import lg

__all__ = ['transfer_entropy', 'tent']


def transfer_entropy(X: List[int], Y: List[int], lag: int = 1) -> float:
    """Compute the transfer entropy from Y to X.
    http://lizier.me/joseph/presentations/20060503-Schreiber-MeasuringInfoTransfer.pdf
    """
    n = len(X)
    bits = 0.0
    xhistory = X[:lag]
    yhistory = Y[:lag]

    xytransition_counts: DefaultDict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], int] = defaultdict(int)
    xyhistory_counts: DefaultDict[Tuple, int] = defaultdict(int)
    xtransition_counts: DefaultDict[Tuple, int] = defaultdict(int)
    xhistory_counts: DefaultDict[Tuple, int] = defaultdict(int)
    xcounts = Counter(X)

    for i, xnext in enumerate(X[lag:]):
        xytransition = (xnext, tuple(xhistory), tuple(yhistory))
        xyhistory = (tuple(xhistory), tuple(yhistory))
        xtransition = (xnext, tuple(xhistory))

        xytransition_counts[xytransition] += 1
        xyhistory_counts[xyhistory] += 1
        xtransition_counts[xtransition] += 1
        xhistory_counts[tuple(xhistory)] += 1

        xhistory = np.delete(xhistory, 0)
        xhistory = np.append(xhistory, xnext)
        yhistory = np.delete(yhistory, 0)
        yhistory = np.append(yhistory, Y[lag + i])

    nxytransition = sum(xytransition_counts.values())
    nxtransition = sum(xtransition_counts.values())
    nxyhistory = sum(xyhistory_counts.values())
    nxhistory = sum(xhistory_counts.values())

    for xytransition, xytransition_count in xytransition_counts.items():
        xnext = xytransition[0]
        xyhist = tuple(xytransition[1:])
        xtrans = tuple(xytransition[:-1])

        pr_xytransition = xytransition_count / nxytransition
        pr_xyhistory = xyhistory_counts[xyhist] / nxyhistory
        pr_xnext_given_xyhistory = pr_xytransition / pr_xyhistory
        pr_xnext = xcounts[xnext] / n
        pr_xtransition = xtransition_counts[xtrans] / nxtransition
        pr_xnext_given_xhistory = pr_xtransition / pr_xnext

        bits += pr_xytransition * \
            (lg(pr_xnext_given_xyhistory) - lg(pr_xnext_given_xhistory))
    return bits


def tent(X: List[int], Y: List[int], max_lag: int) -> Tuple[float, float]:
    tent_YtoX = max(transfer_entropy(X, Y, lag)
                    for lag in range(1, max_lag + 1))
    tent_XtoY = max(transfer_entropy(Y, X, lag)
                    for lag in range(1, max_lag + 1))
    return tent_XtoY, tent_YtoX
