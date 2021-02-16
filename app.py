from ypstruct import structure
import ga

# Assume there's a warehouse which distributes a certain product. The warehouse wishes to maximize the profit ("reward") of the
# distribution. We're given the name, minimum required quantity, maximum required quantity of the produt and distance of each city.
# We're also given the profit for a single amount (e.g. kilogram) of the product ("r"), the penalty ("p", e.g. the price of shippment)
cities = [("Jerusalem", 30, 100, 20), ("Beit Shemesh", 10, 50, 50), ("Tel Aviv", 20, 80, 80), ("Northern Samaria", 0, 50, 200), ("Beer Sheva", 10, 60, 150)]
p = 2 # the penalty
r = 40 # the reward

# Function to be maximized
def f(x):
    sum = 0
    for i in range(len(x)):
        sum += r * x[i] - (cities[i][3] * p) # the total reward for all cities after removing the penalty
    return sum

# Problem Definition
problem = structure()
problem.func = f
problem.nvar = len(cities)
problem.varmin = [c[1] for c in cities]
problem.varmax = [c[2] for c in cities]

# Parameters
params = structure()
params.maxit = 100
params.npop = 50
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
out = ga.run(problem, params, verbose=True)

# Results
print("\nBest Assignment:\n")
best_assignment = out.bestsol.assignment
for i in range(len(best_assignment)):
    print("\t {}: {}".format(cities[i][0], best_assignment[i]))
print("\nBest Reward: {}\n".format(out.bestsol.reward))
ga.plot(params.maxit, out.bestreward)
