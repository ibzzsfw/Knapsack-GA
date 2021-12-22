import numpy as np
from matplotlib import pyplot as plt
from random import choice, choices


def swap(a, b) -> any:
    return b, a


def crossing(a, b, start, stop):

    A, B = list(a), list(b)
    for i in range(start, stop + 1):
        A[i], B[i] = swap(A[i], B[i])

    return ''.join(A), ''.join(B)


def mutate_swap(p: str, i: int, j: int) -> str:

    P = list(p)
    P[i], P[j] = swap(P[i], P[j])

    return ''.join(P)


def split_random(population: np.ndarray):

    pool1 = np.array([], dtype=int)
    index = list(range(len(population)))
    for _ in range(len(population) // 2):
        chosen = choice(index)
        pool1 = np.append(pool1, population[chosen].copy())
        index.remove(chosen)
    pool2 = population[index].copy()

    return pool1, pool2


def flip(bit: str) -> str:

    return '1' if bit == '0' else '0'


class GA:

    def __init__(self, **kwargs): self.__dict__.update(kwargs)

    def initial(self) -> np.array:

        founder = np.array([])

        while len(founder) != self.population_size:
            individual = ''.join(choice('01')for _ in range(self.genome_size))
            founder = np.unique(np.append(founder, individual))

        return founder

    def prime_individual(self):

        ratio = [0 for i in range(self.genome_size)]
        for i in range(self.genome_size):
            ratio[i] = self.knapsack.values[i]/self.knapsack.weights[i]
        ratio_key = sorted(ratio, reverse=True)

        prime_individual = ''.zfill(self.genome_size)
        a = 0
        while self.fitness(prime_individual) >= 0 and a < self.genome_size:
            p = ratio.index(ratio_key[a])
            a += 1
            prime_individual = prime_individual[:p] + flip(prime_individual[p]) + prime_individual[p + 1:]
            if self.fitness(prime_individual) < 0:
                prime_individual = prime_individual[:p] + flip(
                    prime_individual[p]) + prime_individual[p + 1:]
            ratio[p] = 0

        return prime_individual

    def fitness(self, individual: str) -> int:

        # rho = np.max(np.array([self.knapsack.values/self.knapsack.weights]))
        individual_vector = np.array(list(individual), dtype=int)
        # pen = rho * (np.dot(individual_vector, self.knapsack.weights) - self.knapsack.capacity)
        if self.knapsack.capacity < np.dot(individual_vector, self.knapsack.weights):
            # return np.dot(individual_vector, knapsack.values) - np.log2(pen+1)
            return self.knapsack.capacity - np.dot(individual_vector, self.knapsack.weights)
        else:
            return np.dot(individual_vector, self.knapsack.values)

    def selection(self, population: np.ndarray):

        population = population.tolist()
        population.sort(key=lambda p: self.fitness(p), reverse=True)

        return np.array(population[0:self.population_size])

    def crossover(self, population: np.ndarray):

        pool1, pool2 = split_random(population)

        for i in range(len(population)//2):

            if np.random.random_sample() > self.crossover_rate:
                continue

            p1, p2 = pool1[i], pool2[i]

            if self.crossover_method == 'SINGLE_POINT':
                p1, p2 = crossing(p1, p2, 0, choice(range(self.genome_size)))
            elif self.crossover_method == 'TWO_POINT':
                interval = choices(list(range(self.genome_size)), k=2)
                p1, p2 = crossing(p1, p2, min(interval), max(interval))
            elif self.crossover_method == 'UNIFORM':
                uniform = np.random.rand(self.genome_size)
                for j, u in enumerate(uniform):
                    if u >= 0.5:
                        p1, p2 = crossing(p1, p2, j, j)

            population = np.append(population, p1)
            population = np.append(population, p2)

        return population

    def mutation(self, offspring: np.ndarray):

        for p in offspring:
            if np.random.random_sample() > self.mutation_rate:
                continue

            if self.mutation_method == 'BIT_FLIP':
                pb = 1/self.genome_size
                for i in range(self.genome_size):
                    if pb >= np.random.random_sample():
                        p = p[:i] + flip(p[i]) + p[i + 1:]
            elif self.mutation_method == 'SWAP':
                points = np.random.randint(low=0, high=self.genome_size - 1)
                for _ in range(points):
                    point = choices(list(range(self.genome_size)), k=2)
                    p = mutate_swap(p, min(point), max(point))

        return offspring

    def data(self, population):

        return [self.fitness(individual) for individual in population]

    def natural_selection(self):

        import time

        a, b = 0, 1
        avg = np.array([], dtype=float)
        best = np.array([], dtype=int)
        sw = True
        r = True
        population = self.initial()
        prime = self.fitness(self.prime_individual())
        t = time.time()
        population = self.selection(population)

        for i in range(self.generation):
            if a == b or sw == False:
                continue

            print("{}, ".format(i), end='')
            population = self.crossover(population)
            if self.mutation_rate != 0:
                population = self.mutation(population)
            population = self.selection(population)
            gen_data = self.data(population)
            a, b = np.average(gen_data), np.max(gen_data)
            best, avg = np.append(best, b), np.append(avg, a)
            print("{}, {}".format(a, b))
            # termination criteria
            if i > 10 and (prime - b)/prime < 10:
                if best[i] == best[i - 9]:
                    sw = False
            # improvement
            if i > 5 and (prime - b)/prime < 10 and r == True:
                if best[i] == best[i - 4]:
                    self.mutation_rate = 0.8
                    self.crossover_rate = 0.2
                    r = False

        time = time.time() - t
        print("time: {}".format(time))

        # visualize
        X = np.linspace(1, len(avg), len(avg), endpoint=False)
        plt.scatter(X, avg, color='g', label='Average', marker='.')
        plt.scatter(X, best, color='r', label='Maximum', marker='.')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()
