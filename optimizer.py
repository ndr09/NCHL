import math
import torch
from cma import CMAEvolutionStrategy as cmaes
import numpy as np
import functools
from random import Random
import multiprocessing as mp
import torch
import time
from os.path import exists
from os import mkdir


class Best():
    def __init__(self, x=None, f=None):
        self.x = x
        self.f = f

    def __str__(self):
        return str(self.x) + " " + str(self.f)


# Vanilla LMMAES algorithm
class LMMAES(object):
    """
    LMMAES algorithm: this algorithm implements the LMMAES algorithm
    Low Memory Matrix Adaptation Evolution Strategy proposed in the paper:
    "Large Scale Black-Box Optimization by Limited-Memory Matrix Adaptation"
    Here is the link: https://ieeexplore.ieee.org/document/8410043
    """

    def __init__(
            self,
            n,
            seed,
            lambda_=None,
            mu=None,
            m=None,
            sigma=None,
            device='cpu',
    ):
        """
        Initialize the LMMAES algorithm
        :param n: number of dimensions of the problem
        :param lambda_: number of generated offsprings
        :param mu: number of selected individuals
        :param m: number of vectors that will approximate the covariance matrix
        :param sigma: learning rate
        :param device: device to use for torch
        """
        # device to use
        self.device = device
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        # number of parameters
        self.n = n

        # number of generated offsprings
        # default: 4 + 3*ln(n)
        self.lambda_ = lambda_ if lambda_ is not None else 4 + int(math.floor(3 * math.log(n)))

        # number of selected individuals
        # default: lambda/2
        self.mu = mu if mu is not None else int(math.floor(self.lambda_ / 2))

        # weight vector initialization assigned to each selected individual
        # default: log(mu+1/2)-log(i) for i=1,...,mu
        denominator = sum([math.log(self.mu + 1 / 2) - math.log(j + 1) for j in range(self.mu)])
        self.w = torch.tensor([(math.log(self.mu + 1 / 2) - math.log(i + 1)) / denominator for i in range(self.mu)]).to(
            self.device)

        # mu_w vector initialization
        # weight assigned to all selected individual
        self.mu_w = 1 / torch.sum(self.w ** 2)

        # m parameter initialization -> default 4 + 3*ln(n)
        # number of vectors that will approximate the covariance matrix
        self.m = 4 + int(math.floor(3 * math.log(n))) if m is None else m

        # c_sigma initialization -> default 2*lambda/n
        # parameter for the Cumulative Step Size Adaptation
        # controls the learning rate of the step size adaptation
        self.c_sigma = (self.lambda_ * 2) / self.n

        # c_d initialization
        # it is a weight vector exponentially decaying
        # to appy on every vector approximating the matrix M
        self.c_d = torch.tensor([1 / ((1.5 ** i) * self.n) for i in range(self.m)]).to(self.device)

        # c_c initialization
        self.c_c = torch.tensor([self.lambda_ / ((4 ** i) * self.n) for i in range(self.m)]).to(self.device)

        # init centroid vector
        # init to a zero vector
        self.y = torch.randn(self.n, generator=self.rng).float().to(device)

        # init sigma, this is my global learning rate
        self.sigma = sigma if sigma is not None else 1.0

        # init the evolution path vector p_sigma
        # it is an exponentially fading record of recent most successful steps
        self.p_sigma = torch.zeros(self.n).float().to(self.device)

        # init the vector estimating the covariance matrix
        self.M = torch.zeros((self.m, self.n)).float().to(self.device)

        # init vectors containing the offspring's direction vectors
        self.d = torch.zeros((self.lambda_, self.n)).float().to(self.device)

        # init vectors containing the offspring's randomness vectors
        self.z = torch.zeros((self.lambda_, self.n)).float().to(self.device)

        # init the number of iterations
        self.t = 0
        self.population = []
        self.best = Best()

    def ask(self):
        """
        Generate a new population of individuals
        :return: the new population
        """
        # z are lambda samples from a normal distribution
        self.population = []
        for i in range(self.lambda_):
            self.z[i] = self.create_z()
            # direction vector -> initialized as random
            self.d[i] = self.z[i].clone()
            # direction vector is updated with the previous m directions
            for j in range(min(self.t, self.m)):
                # if d and M has the same direction, similarity is high,
                # it means that the direction is good, and we can use
                # it to update the direction vector by that factor
                similarity = (self.M[j] @ self.d[i])
                self.d[i] = (1 - self.c_d[j]) * self.d[i] + self.c_d[j] * self.M[j] * similarity

            # creating the individual
            # d[i] is now the mutation given by N(0, C) where C is the covariance matrix
            ind = (self.y + self.sigma * self.d[i]).detach()

            self.population.append(ind.tolist())
        return self.population[:]

    def get_sorted_idx(self, fitness):
        """
        Get the ordered list of the indexes of the individuals, ordered by fitness
        :param population: population of individuals
        :return: the ordered list of the indexes of the individuals, ordered by fitness
        """
        # get the ordered list of the indexes of the mu best individuals
        sorted_idx = [i for _, i in sorted(zip(fitness, range(len(fitness))))][0:self.mu]
        return sorted_idx

    def tell(self, fitness):
        """
        Update the parameters of the algorithm
        :param fitness: fitness of the individual in the same order
        :param population: generated population, already evaluated
        """

        # get the ordered list of the indexes of the mu best individuals
        sorted_idx = self.get_sorted_idx(fitness)
        if self.best.x is None or self.best.f < fitness[sorted_idx[0]]:
            self.best = Best(self.population[sorted_idx[0]], fitness[sorted_idx[0]])
        # calculate the weighted sum of the mu best individuals
        weighted_d = torch.zeros((self.mu, self.n)).float().to(self.device)
        weighted_z = torch.zeros((self.mu, self.n)).float().to(self.device)

        j = 0
        for i in sorted_idx:
            weighted_d[j] = self.w[j] * self.d[i]
            weighted_z[j] = self.w[j] * self.z[i]
            j += 1

        # update the evolution path of the best solutions
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma \
                       + torch.sqrt(self.mu_w * self.c_sigma * (2 - self.c_sigma)) \
                       * torch.sum(weighted_z, dim=0)

        # update the support vectors for the covariacne matrix
        for i in range(self.m):
            self.M[i] = (1 - self.c_c[i]) * self.M[i] \
                        + torch.sqrt(self.mu_w * self.c_c[i] * (2 - self.c_c[i])) \
                        * torch.sum(weighted_z, dim=0)

        # update sigma
        self.sigma = self.sigma * torch.exp(((torch.norm(self.p_sigma) ** 2 / self.n) - 1) * self.c_sigma / 2)

        # calculate new centroid
        self.y = self.y + self.sigma * torch.sum(weighted_d, dim=0)

        # update the number of iterations
        self.t += 1

    def create_z(self):
        """
        Create a new noise vector
        :return: the noise vector
        """
        return torch.randn(self.n, generator=self.rng).float().to(self.device)


def generator(random, args):
    return np.asarray([random.uniform(args["pop_init_range"][0],
                                      args["pop_init_range"][1])
                       for _ in range(args["num_vars"])])


def generator_wrapper(func):
    @functools.wraps(func)
    def _generator(random, args):
        return np.asarray(func(random, args))

    return _generator


class CMAES():

    def __init__(self, num_vars, seed, pop_init_range, lmbda, mu, sigma):
        args = {
            "num_vars": num_vars,
            "pop_init_range": pop_init_range,
        }
        self.cmaes = cmaes(generator(Random(seed), args),
                           sigma,
                           {'popsize': lmbda,
                            'seed': seed,
                            'CMA_mu': mu})
        self.pop = []
        self.best = None

    def tell(self, fitness):
        self.cmaes.tell(self.pop, fitness)
        self.best = self.cmaes.best

    def ask(self):
        self.pop = self.cmaes.ask()
        return self.pop[:]


class ES1():
    def __init__(self, seed, length, npop, sigma):
        self.seed = seed
        self.length = length
        self.npop = npop
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.fitness = np.zeros(npop)
        self.elite = None
        self.parents = []

    def ask(self):
        if self.elite is None:
            return self.rng.uniform(-1., 1., (self.npop, self.length))
        else:
            sums = np.mean(self.parents, axis=0)

            ns = []
            for i in range(self.npop - 1):
                a = sums + self.rng.standard_normal(size=self.length) * self.sigma
                ns.append(a)

            ns.append(self.elite[1])
            return ns

    def tell(self, pop, fits):
        elites = [(fits[i], pop[i]) for i in range(self.npop)]
        self.fitness = fits
        elites.sort(reverse=True, key=lambda t: t[0])

        if self.elite is None or elites[0][0] > self.elite[0]:
            self.elite = elites[0]
        self.parents = np.array([t[1] for t in elites[:10]])


class EvolutionStrategy(object):
    """
    This code is slightly modified from this code:
        https://github.com/enajx/HebbianMetaLearning/blob/master/evolution_strategy_hebb.py
    """

    def __init__(self, seed, n_params, init_weights='uni', population_size=100, sigma=0.1,
                 learning_rate=0.2,
                 decay=0.995, distribution='normal'):

        self.n_params = n_params
        self.init_weights = init_weights
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.distribution = distribution

        self._npops = []
        self._npops_coev = []
        self.rng = np.random.default_rng(seed)
        self.trng = torch.Generator()
        self.trng.manual_seed(seed)
        self.best = Best()
        if self.distribution == 'uniform':
            self.coeffs = self.rng.uniform(-1, 1, self.n_params)
        elif self.distribution == 'normal':
            self.coeffs = torch.randn(self.n_params, generator=self.trng).detach().numpy().squeeze()

    def _get_params_try(self, w, p):

        param_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            param_try.append(w[index] + jittered)
        param_try = np.array(param_try).astype(np.float32)

        return param_try
        # return w + p*self.SIGMA

    def get_coeffs(self):
        return self.coeffs.astype(np.float32)

    def _compute_ranks(self, x):
        """
        Returns rank as a vector of len(x) with integers from 0 to len(x)
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)

        ranks[x.argsort()] = np.arange(len(x))

        return ranks

    def _compute_centered_ranks(self, x):
        """
        Maps x to [-0.5, 0.5] and returns the rank
        """
        y = self._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y

    def _get_population(self):

        # x_ = np.random.randn(int(self.POPULATION_SIZE/2), self.coeffs.shape[0], self.coeffs[0].shape[0])
        # population = np.concatenate((x_,-1*x_)).astype(np.float32)

        population = []

        for i in range(int(self.POPULATION_SIZE / 2)):
            x = []
            x2 = []
            for w in self.coeffs:
                j = self.rng.standard_normal(*w.shape)  # j: (coefficients_per_synapse, 1) eg. (5,1)
                x.append(j)  # x: (coefficients_per_synapse, number of synapses) eg. (92690, 5)
                x2.append(-j)
            population.append(
                x)  # population : (population size, coefficients_per_synapse, number of synapses), eg. (10, 92690, 5)
            population.append(x2)

        return np.array(population).astype(np.float32)

    def ask(self):
        pop = self._get_population()
        self._npops = [self._get_params_try(self.coeffs, p) for p in pop]
        return self._npops

    def _update_coeffs(self, rewards, population):
        rewards = self._compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')

        rewards = (rewards - rewards.mean()) / std

        for index, c in enumerate(self.coeffs):
            layer_population = np.array([p[index] for p in population])

            self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.coeffs[index] = c + self.update_factor * np.dot(layer_population.T, rewards).T

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        # Decay sigma
        if self.SIGMA > 0.01:
            self.SIGMA *= 0.999

    def tell(self, fitness):
        # print(fitness)
        fitness = np.array(fitness)
        # print(fitness)

        newBestIndex = np.argmax(fitness)
        if self.best.f is None or fitness[newBestIndex] > self.best.f:
            self.best = Best(self._npops[newBestIndex], fitness[newBestIndex])
        self._update_coeffs(fitness, self._npops)
