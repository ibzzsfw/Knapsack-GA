from genetic import GA
from knapsack import Knapsack
import numpy as np


if __name__ == "__main__":

    PATH = "data/E.json"
    knapsack = Knapsack(PATH)
    print("File loaded: {}".format(PATH))

    PARAMETER = {
        'knapsack': knapsack,
        'generation': 2000, #customizable
        'genome_size': knapsack.n,
        'population_size': int((np.pi**2+np.pi)*np.sqrt(knapsack.n)), #customizable
        'crossover_method': 'UNIFORM', # CROSSOVER METHOD: SINGLE_POINT, TWO_POINT, UNIFORM
        'crossover_rate': 0.8, #customizable
        'mutation_method': 'BIT_FLIP', # MUTATION METHOD:  BIT_FLIP, SWAP
        'mutation_rate': 0.1, #customizable
    }

    GA(**PARAMETER).natural_selection()
