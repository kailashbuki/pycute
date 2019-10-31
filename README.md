Pycute
-------

Pycute is a infromation-theoretic causal inference method for event sequences based on Granger-causality.

Pycute Module Installation
----------------------------

The recommended way to install the `pycute` module is to simply use `pip`:

```console
$ pip install pycute
```
Pycute officially supports Python >= 3.6.

How to use pycute?
------------------
```pycon
>>> X = [1] * 1000
>>> Y = [-1] * 1000
>>> from pycute import cute, tent, simulations
>>> cute.cute(X, Y)                                                   # CUTE
(0.0, 0.0)
>>> tent.tent(X, Y)                                                   # TENT
(0.0, 0.0)
>>> simulations.simulate_decision_rate_against_data_type('/results/dir/')
# for decision rate vs causal relationship type plots
...
```

How to cite the paper?
----------------------
Todo: Add the citation to thesis.
