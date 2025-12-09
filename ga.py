"""
Genetic Algorithm (GA) Implementation
Based on evolutionary principles and natural selection
"""

import numpy as np
import time


class Solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiters = 0


class GeneticAlgorithm:
    """
    Genetic Algorithm for continuous optimization
    
    Parameters:
    -----------
    fitness_func : callable
        The objective function to minimize
    dim : int
        Problem dimension
    pop_size : int
        Population size (number of chromosomes)
    max_iter : int
        Maximum number of generations
    bounds : tuple
        Search space bounds (lb, ub)
    crossover_rate : float
        Probability of crossover (default: 0.8)
    mutation_rate : float
        Probability of mutation (default: 1/dim)
    tournament_size : int
        Tournament selection size (default: 3)
    """
    
    def __init__(self, fitness_func, dim, pop_size=30, max_iter=1000, 
                 bounds=(-100, 100), crossover_rate=0.8, mutation_rate=None,
                 tournament_size=3):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        if isinstance(bounds, tuple):
            self.lb = np.ones(dim) * bounds[0]
            self.ub = np.ones(dim) * bounds[1]
        else:
            self.lb = bounds[0]
            self.ub = bounds[1]
            
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate if mutation_rate else 1.0 / dim
        self.tournament_size = tournament_size
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = np.inf
        self.convergence_curve = []
        
    def initialize_population(self):
        """Initialize population randomly within bounds"""
        population = np.random.uniform(
            low=self.lb, 
            high=self.ub, 
            size=(self.pop_size, self.dim)
        )
        return population
    
    def evaluate_population(self, population):
        """Evaluate fitness for entire population"""
        fitness = np.array([self.fitness_func(ind) for ind in population])
        return fitness
    
    def tournament_selection(self, population, fitness):
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(
            self.pop_size, 
            self.tournament_size, 
            replace=False
        )
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def simulated_binary_crossover(self, parent1, parent2, eta=20):
        """
        Simulated Binary Crossover (SBX)
        
        Parameters:
        -----------
        parent1, parent2 : np.ndarray
            Parent chromosomes
        eta : float
            Distribution index (larger values give children closer to parents)
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros(self.dim)
        child2 = np.zeros(self.dim)
        
        for j in range(self.dim):
            if np.random.rand() <= 0.5:
                if abs(parent1[j] - parent2[j]) > 1e-14:
                    u = np.random.rand()
                    
                    if u <= 0.5:
                        beta_q = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    child1[j] = 0.5 * ((1 + beta_q) * parent1[j] + (1 - beta_q) * parent2[j])
                    child2[j] = 0.5 * ((1 - beta_q) * parent1[j] + (1 + beta_q) * parent2[j])
                else:
                    child1[j] = parent1[j]
                    child2[j] = parent2[j]
            else:
                child1[j] = parent1[j]
                child2[j] = parent2[j]
        
        # Boundary check
        child1 = np.clip(child1, self.lb, self.ub)
        child2 = np.clip(child2, self.lb, self.ub)
        
        return child1, child2
    
    def polynomial_mutation(self, individual, eta_m=20):
        """
        Polynomial mutation
        
        Parameters:
        -----------
        individual : np.ndarray
            Individual to mutate
        eta_m : float
            Mutation distribution index
        """
        mutant = individual.copy()
        
        for j in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                u = np.random.rand()
                delta_1 = (mutant[j] - self.lb[j]) / (self.ub[j] - self.lb[j])
                delta_2 = (self.ub[j] - mutant[j]) / (self.ub[j] - self.lb[j])
                
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if u <= 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                mutant[j] = mutant[j] + delta_q * (self.ub[j] - self.lb[j])
                mutant[j] = np.clip(mutant[j], self.lb[j], self.ub[j])
        
        return mutant


def GA(objf, lb, ub, dim, N, Max_time, crossover_rate=0.8, mutation_rate=None, tournament_size=3):
    """
    Genetic Algorithm
    
    Parameters:
    -----------
    objf : function
        Objective function to minimize
    lb : float or list
        Lower bound(s)
    ub : float or list
        Upper bound(s)
    dim : int
        Problem dimension
    N : int
        Population size
    Max_time : int
        Maximum iterations (generations)
    crossover_rate : float
        Probability of crossover (default: 0.8)
    mutation_rate : float
        Probability of mutation (default: 1/dim)
    tournament_size : int
        Tournament selection size (default: 3)
    
    Returns:
    --------
    s : Solution
        Solution object with results
    """
    
    # Handle bounds
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Set mutation rate
    if mutation_rate is None:
        mutation_rate = 1.0 / dim
    
    # Initialize population
    population = np.random.uniform(low=lb, high=ub, size=(N, dim))
    
    # Initialize best
    Best_chromosome = np.zeros(dim)
    Best_fitness = float("inf")
    
    convergence = np.zeros(Max_time)
    
    s = Solution()
    
    print(f'GA is optimizing "{objf.__name__}"')
    
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main evolution loop
    for iteration in range(Max_time):
        
        # Evaluate population
        fitness = np.zeros(N)
        for i in range(N):
            # Apply bounds
            population[i, :] = np.clip(population[i, :], lb, ub)
            
            # Evaluate
            fitness[i] = objf(population[i, :])
            
            # Update best
            if fitness[i] < Best_fitness:
                Best_fitness = fitness[i]
                Best_chromosome = population[i, :].copy()
        
        # Store convergence
        convergence[iteration] = Best_fitness
        
        # Generate new population
        new_population = []
        
        while len(new_population) < N:
            # Tournament selection
            parent1_idx = np.random.choice(N, tournament_size, replace=False)
            parent1 = population[parent1_idx[np.argmin(fitness[parent1_idx])]].copy()
            
            parent2_idx = np.random.choice(N, tournament_size, replace=False)
            parent2 = population[parent2_idx[np.argmin(fitness[parent2_idx])]].copy()
            
            # Simulated Binary Crossover (SBX)
            if np.random.rand() <= crossover_rate:
                eta = 20
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                
                for j in range(dim):
                    if np.random.rand() <= 0.5:
                        if abs(parent1[j] - parent2[j]) > 1e-14:
                            u = np.random.rand()
                            
                            if u <= 0.5:
                                beta_q = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta_q = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            
                            child1[j] = 0.5 * ((1 + beta_q) * parent1[j] + (1 - beta_q) * parent2[j])
                            child2[j] = 0.5 * ((1 - beta_q) * parent1[j] + (1 + beta_q) * parent2[j])
                        else:
                            child1[j] = parent1[j]
                            child2[j] = parent2[j]
                    else:
                        child1[j] = parent1[j]
                        child2[j] = parent2[j]
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Polynomial mutation
            eta_m = 20
            for child in [child1, child2]:
                for j in range(dim):
                    if np.random.rand() < mutation_rate:
                        u = np.random.rand()
                        delta_1 = (child[j] - lb[j]) / (ub[j] - lb[j])
                        delta_2 = (ub[j] - child[j]) / (ub[j] - lb[j])
                        
                        mut_pow = 1.0 / (eta_m + 1.0)
                        
                        if u <= 0.5:
                            xy = 1.0 - delta_1
                            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
                            delta_q = val ** mut_pow - 1.0
                        else:
                            xy = 1.0 - delta_2
                            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
                            delta_q = 1.0 - val ** mut_pow
                        
                        child[j] = child[j] + delta_q * (ub[j] - lb[j])
                        child[j] = np.clip(child[j], lb[j], ub[j])
            
            # Apply bounds
            child1 = np.clip(child1, lb, ub)
            child2 = np.clip(child2, lb, ub)
            
            new_population.append(child1)
            if len(new_population) < N:
                new_population.append(child2)
        
        # Replace population
        population = np.array(new_population[:N])
        
        # Print progress
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{Max_time}: Best fitness = {Best_fitness:.6e}")
    
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "GA"
    s.bestIndividual = Best_chromosome
    s.best = Best_fitness
    s.objfname = objf.__name__
    s.lb = lb
    s.ub = ub
    s.dim = dim
    s.popnum = N
    s.maxiters = Max_time
    
    return s


# Test functions
def sphere(x):
    """Sphere function - f(0) = 0"""
    return np.sum(x**2)


def rastrigin(x):
    """Rastrigin function - f(0) = 0"""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing GA on Sphere Function")
    print("="*60)
    
    result = GA(sphere, lb=-100, ub=100, dim=10, N=50, Max_time=100)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Best fitness: {result.best:.6e}")
    print(f"Execution time: {result.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result.bestIndividual[:5]}")
