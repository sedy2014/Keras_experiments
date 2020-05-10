from __future__ import print_function
import numpy as np
from hyperopt import fmin, tpe, hp,Trials
import matplotlib.pyplot as plt
import pandas as pd
def objective(x):
    """Objective function to minimize"""

    # Create the polynomial object
    # x^6 - 2x^5 ...
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Return the value of the polynomial at X specified
    return f(x) * 0.05

xp = np.linspace(-5,6,100)
y = objective(xp)
# plt.plot(xp,y)
# plt.show()

# Create the domain space
#space = hp.uniform('x', -5, 6)
space = hp.normal('x', -5, 6)
# Optimization Algorithm
from hyperopt import tpe

# Create the algorithm
tpe_algo = tpe.suggest

#if we want to inspect the progression of the alogorithm,
# we need to create a Trials object that will record the values and the scores:

tpe_trials = Trials()

# Run 2000 evals with the tpe algorithm
tpe_best = fmin(fn=objective, space=space,
                algo=tpe_algo, trials=tpe_trials,
                max_evals=2000)

print(tpe_best)
#We can see that initially the algorithm picks values from the whole range equally (uniformly), but as time goes on and more is learned about the parameter’s effect on the objective function,
# the algorithm focuses more and more on areas in which it thinks it will gain the most

f, ax = plt.subplots(1)
xs = [t['tid'] for t in tpe_trials.trials]
ys = [t['misc']['vals']['x'] for t in tpe_trials.trials]
ax.set_xlim(xs[0]-10, xs[-1]+10)
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
ax.set_xlabel('$t$', fontsize=16)
ax.set_ylabel('$x$', fontsize=16)
plt.show()

f, ax = plt.subplots(1)
xs = [t['misc']['vals']['x'] for t in tpe_trials.trials]
ys = [t['result']['loss'] for t in tpe_trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$val$', fontsize=16)
plt.show()