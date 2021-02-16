from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
from math import inf

price_of_gas_per_km = 2

def objective_function(x, rewards, distances):
    sum = 0
    for i in range(len(rewards)):
        for j in range(len(distances)):
            sum += x[i * len(distances) + j] * rewards[i] - (distances[j] * price_of_gas_per_km)
    return -sum


r = [20, 60, 100]
d = [60, 20, 100, 200]
x = np.zeros(shape=(len(r) * len(d)))
res = minimize(
    objective_function,
    x0=x,
    args=(r, d,),
    constraints=NonlinearConstraint(lambda x: sum([x[i] for i in range(len(x)) if i % 4 == 0]), lb=9, ub=9),
    bounds=[(0, 5) for n in range(len(r) * len(d))],
    )
print(res)
print(res.x[0] + res.x[4] + res.x[8])