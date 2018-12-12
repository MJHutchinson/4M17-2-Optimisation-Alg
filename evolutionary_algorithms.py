import numpy as np

DIM = 5
EVAL_LIM = 10000
BURN_IN = 150
SIGMA_INIT = 50

def gen_initial_population(population_size=150):
    population = []

    sigma = np.ones(DIM) * SIGMA_INIT

    for i in range(population_size):
        population


def search():

    # initialise population with random samples
    population = gen_initial_population(BURN_IN)


if __name__ == '__main--':
    print("")