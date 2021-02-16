import matplotlib.pyplot as plt
import numpy as np
from ypstruct import structure

# Given the number of iterations and the best reward value found in each iteration,
# plot the best rewards as a function of the iteration.
def plot(num_iterations, iterations_best_rewards):
    plt.plot(iterations_best_rewards)
    plt.xlim(0, num_iterations)
    plt.xlabel('Iterations')
    plt.ylabel('Best Reward')
    plt.title('Genetic Algorithm (GA)')
    plt.grid(True)
    plt.show()

# Run the Genetic Algorithm
#
# Accepts 2 ypstruct.structure objects:
#
# problem:
#   .func   // Maximization Objective Function (should accept a single numpy array of nvar size)
#   .nvar   // Number of Variables
#   .varmin // Min Values for Variables
#   .varmax // Max Values for Variables
#
# params:
#   .maxit  // Maximum Number of Iterations
#   .npop   // Size of Population
#   .beta   // Exponentation Parameter for Generating Chromosome Selection Probabilities for Roulette Wheel Selection
#   .pc     // Population Count Increase Ratio
#   .gamma  // Width of the Interval from Which the Crossover "Absorption" Rates are Drawn Uniformly from
#   .mu     // Probability In Which Mutation Will be Done to Each Coordinate in Offspring Chromosomes
#   .sigma  // Scale Value for the Standard Distribution (mean = 0, var = sigma ** 2) from Which Mutation Values Will be Drawn
#
# Returns a ypstruct.structure object:
#
# out:
#   .pop        // The Population At the Final Iteration (each element is a solution like bestsol)
#   .bestsol:   // The Best Solution At the Final Iteration
#       .assignment // The Assignment of Variables For the Solution
#       .reward     // The Reward Generated For the Assignment Of the Solution
#   .bestreward // A numpy array Containing the Best Rewards at Each Iteration
def run(problem, params, verbose=False):
    # Problem Information
    func = problem.func
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    # Given the Population Count Increase Ratio, Calculate How Many Crossovers Will be Done In Each Iteration.
    # Let's also make sure it will always be an even Integer (so it will allign elegantly with the integer division
    # done in the crossover loop)
    nc = int(np.round(pc * npop / 2) * 2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.assignment = None
    empty_individual.reward = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.reward = -np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].assignment = np.random.uniform(varmin, varmax, nvar)
        pop[i].reward = func(pop[i].assignment)
        #  bestsol Needs to be Replaced.
        if pop[i].reward > bestsol.reward:
            bestsol = pop[i].deepcopy()

    # Best Costs of Iterations
    bestreward = np.empty(maxit)

    # Main Loop
    for it in range(maxit):
        rewards = np.array([x.reward for x in pop])
        avg_reward = np.mean(rewards)
        if avg_reward != 0:
            rewards = rewards / avg_reward
        probs = np.exp(-beta * rewards)

        popc = []
        for _ in range(nc // 2):
            # Perform Roulette Wheel Selection
            p1 = pop[__roulette_wheel_selection(probs)]
            p2 = pop[__roulette_wheel_selection(probs)]

            # Perform Crossover
            c1, c2 = __crossover(p1, p2, gamma)

            # Perform Mutation
            c1 = __mutate(c1, mu, sigma)
            c2 = __mutate(c2, mu, sigma)

            # Apply Bounds
            __apply_bound(c1, varmin, varmax)
            __apply_bound(c2, varmin, varmax)

            # Evaluate First Offspring
            c1.reward = func(c1.assignment)
            if c1.reward > bestsol.reward:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.reward = func(c2.assignment)
            if c2.reward > bestsol.reward:
                bestsol = c2.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # Merge
        pop += popc
        # Sort
        pop = sorted(pop, key=lambda x: x.reward, reverse=True)
        # Select
        pop = pop[0:npop]

        # Store Best Reward
        bestreward[it] = bestsol.reward

        if verbose:
            # Show Iteration Information
            print("Iteration {}: Best Reward in Iteration = {}".format(it, bestreward[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestreward = bestreward
    return out

# Given 2 chromosomes, croosvover is done as follows:
#
# Uniformly draw n values (this vector is alpha) in [-gamma, 1 + gamma], where n is the number of variables in the problem.
# Then, the 1st child will be a alpha * 1st parent + (1 - alpha) * 2nd parent,
# and the 2nd child will be alpha * 2nd parent + (1 - alpha) * 1st parent.
def __crossover(p1, p2, gamma):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma, 1 + gamma, *c1.assignment.shape)
    c1.assignment = alpha * p1.assignment + (1 - alpha) * p2.assignment
    c2.assignment = alpha * p2.assignment + (1 - alpha) * p1.assignment
    return c1, c2

# Given Chromosme, mutate it is as follows:
#
# Uniformly draw n values from [0, 1], where n is is the number of variables in the problem.
# Then, for each value, check if it is <= mu. The variables at the indexes where the condition is true will
# be incrementerd by a Uniformly distributed random value in [0, 1].
def __mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.assignment.shape) <= mu
    ind = np.argwhere(flag)
    y.assignment[ind] += sigma * np.random.randn(*ind.shape)
    return y

# Apply the given max and min bounds to the variables of the given chromosome.
def __apply_bound(x, varmin, varmax):
    x.assignment = np.maximum(x.assignment, varmin)
    x.assignment = np.minimum(x.assignment, varmax)

# Roulette Wheel Selection is implemented as follows:
#
# Calculate the cummulative sum for the given probabilites ("p"), then uniformly draw a ramndom
# value in [0, sum(p)] and return the 1st index where the cummulative sum is bigger than that value.
def __roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
