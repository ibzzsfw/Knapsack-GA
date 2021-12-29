import json
import numpy as np


class Knapsack:

    def __init__(self, path: str) -> none:

        with open(path, 'r') as file:
            self.__dict__ = json.load(file)
            
        self.weights = np.array(self.weights, dtype=np.int)
        self.values = np.array(self.values, dtype=np.int)
