#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Tests on synthetic data are here.
"""
from collections import defaultdict
from enum import Enum
import os
import random
from typing import DefaultDict, List, Tuple

import numpy as np                  # type: ignore
import pandas as pd                 # type: ignore
from tqdm import tqdm   # type: ignore

from .cute import cute, regret
from .tent import tent


__all__ = ['simulate_monotonicity', 'simulate_decision_rates']


class FuncType(Enum):
    SHIFTED: int = 1
    SHIFTED_INVERTED: int = 2
    RULE_BASED = 3


def _gen_XY(func_type: FuncType, size: int) -> Tuple[List[int], List[int]]:
    if func_type == FuncType.SHIFTED or func_type == FuncType.SHIFTED_INVERTED:
        shift_by = random.randint(1, 20)
        X = _gen_X(size, shift_by)
        Y = _shift(X, shift_by, func_type == FuncType.SHIFTED_INVERTED)
        # we intentionally generate X of length=shift_by+size
        # thus we have to remove the elements from the start till shift_by
        X = X[shift_by:]
    elif func_type == FuncType.RULE_BASED:
        X = _gen_X(size)
        Y = _transform_by_rule(X)
    else:
        raise ValueError('Unknown func_type `%s`.' % str(func_type))
    return X, Y


def _dr_curve(decisions: np.array,
              score_diffs: np.array) -> Tuple[List[float], List[float]]:
    num_pairs = np.size(score_diffs)
    decision_rates, accuracies = [], []

    unq_score_diffs = np.unique(score_diffs)[::-1]
    for score_diff in unq_score_diffs:
        ok_pairs = score_diffs >= score_diff
        decision_ok_pairs = decisions[ok_pairs]

        num_ok_pairs = np.sum(ok_pairs)
        decision_rate = num_ok_pairs / num_pairs
        ncorrect_decisions = np.sum(decision_ok_pairs)
        accuracy = ncorrect_decisions / num_ok_pairs

        accuracies.append(accuracy)
        decision_rates.append(decision_rate)
    return decision_rates, accuracies


def _gen_X(size: int, shift_by: int = 0) -> List[int]:
    seq = []
    p_one = random.uniform(0.1, 0.5)
    for i in range(size + shift_by):
        next = 1 if random.random() < p_one else 0
        seq.append(next)
    assert len(seq) == size + shift_by
    return seq


def _shift(seq: List[int], shift_by: int, invert: bool = False) -> List[int]:
    # forward shift
    size = len(seq) - shift_by
    shifted_seq = [0] * size
    for i in range(size):
        bit = seq[i] if not invert else 1 ^ seq[i]
        shifted_seq[i] = bit
    return shifted_seq


def _transform_by_rule(seq: List[int]) -> List[int]:
    transformed_seq = []
    for i in range(len(seq)):
        if not i:
            transformed_seq.append(random.randint(0, 1))
            continue
        c = seq[i - 1]
        e = transformed_seq[i - 1]
        if c == e:
            new_element = random.randint(0, 1)
        elif c == 0 and e == 1:
            new_element = 0
        else:
            new_element = 1
        transformed_seq.append(new_element)
    assert len(transformed_seq) == len(seq)
    return transformed_seq


def _validate_dir(dirpath: str) -> None:
    if not os.path.isdir(dirpath):
        raise IOError('Invalid directory: %s' % dirpath)


def _add_noise(value: int, noise: float) -> int:
    return value ^ 1 if random.random() < noise else value


def _run_cute(X: List[int], Y: List[int]) -> Tuple[bool, bool, float]:
    delta_XtoY, delta_YtoX = cute(X, Y)
    decision = delta_XtoY != delta_YtoX
    XtoY = delta_XtoY > delta_YtoX
    diff = abs(delta_XtoY - delta_YtoX)
    return decision, XtoY, diff


def _run_tent(X: List[int], Y: List[int], max_lag: int) -> Tuple[bool, bool, float]:
    tent_XtoY, tent_YtoX = tent(X, Y, max_lag)
    decision = tent_XtoY != tent_YtoX
    XtoY = tent_XtoY > tent_YtoX
    diff = abs(tent_XtoY - tent_YtoX)
    return decision, XtoY, diff


def simulate_monotonicity(results_dir: str, to_predict: int) -> None:
    _validate_dir(results_dir)

    tt = range(50)      # x
    t1t1 = range(50)    # y
    fname = "pone.dat" if to_predict == 1 else "pzero.dat"
    fpath = os.path.join(results_dir, fname)
    fp = open(fpath, "w")
    fp.write("t1\tt\tregret\n")
    for t in tt:
        for t1 in t1t1:
            res = regret(t1, t, to_predict) if t1 < t else float("nan")
            fp.write("%d\t%d\t%.4f\n" % (t1, t, res))
        fp.write("\n")
    fp.close()


def simulate_decision_rates(results_dir: str,
                            nsample: int = 200,
                            sample_size=1000,
                            max_lag: int = 20) -> None:
    _validate_dir(results_dir)

    noises = np.arange(0.0, 0.35, 0.1)
    func_types = [f for f in FuncType]
    for func_type in tqdm(func_types):
        decs_by_method: DefaultDict[str, List[bool]] = defaultdict(list)
        diffs_by_method: DefaultDict[str, List[float]] = defaultdict(list)

        for i in range(nsample):
            X, Y = _gen_XY(func_type, sample_size)
            assert len(X) == len(Y)
            noise = np.random.choice(noises)
            Y = [_add_noise(y, noise) for y in Y]

            cute_dec, cute_XtoY, cute_diff = _run_cute(X, Y)
            tent_dec, tent_XtoY, tent_diff = _run_tent(X, Y, max_lag)

            if cute_dec:
                decs_by_method['cute'].append(cute_XtoY)
                diffs_by_method['cute'].append(cute_diff)
            if tent_dec:
                decs_by_method['tent'].append(tent_XtoY)
                diffs_by_method['tent'].append(tent_diff)

        for method in decs_by_method:
            fname = '%s-drate-%s.csv' % (str(func_type), method)
            rates, accs = _dr_curve(np.array(decs_by_method[method]),
                                    np.array(diffs_by_method[method]))
            df = pd.DataFrame(data=dict(rates=rates, accuracies=accs))
            df.to_csv(os.path.join(results_dir, fname), sep=',', index=False)
