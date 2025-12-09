"""
Gravitational Search Algorithm (GSA) Implementation
Based on Newton's law of gravity and motion

Particles attract each other based on their mass (fitness)
Better solutions have higher mass and attract others more strongly
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


def GSA(objf, lb, ub, dim, N, Max_time, G0=100, alpha=20):
    """
    Gravitational Search Algorithm (GSA)
    
    Based on Newton's law of gravitation:
    - Particles attract each other based on their mass (fitness)
    - Better solutions have higher mass
    - Gravitational constant G decreases over time
    
    Parameters:
    -----------
    objf : function
        Objective function to minimize
    lb : float or array
        Lower bound(s)
    ub : float or array
        Upper bound(s)
    dim : int
        Problem dimension
    N : int
        Population size (number of agents)
    Max_time : int
        Maximum number of iterations
    G0 : float, optional
        Initial gravitational constant (default: 100)
    alpha : float, optional
        Gravitational constant decay rate (default: 20)
    
    Returns:
    --------
    Solution object with best fitness, best individual, and convergence history
    """
    
    # Initialize solution object
    s = Solution()
    s.optimizer = "GSA"
    s.objfname = objf.__name__
    s.startTime = time.time()
    s.lb = lb
    s.ub = ub
    s.dim = dim
    s.popnum = N
    s.maxiters = Max_time
    
    # Random number generator
    rng = np.random.default_rng()
    
    # Initialize positions and velocities
    positions = rng.uniform(lb, ub, (N, dim))
    velocities = np.zeros((N, dim))
    
    # Evaluate initial population
    fitness = np.array([objf(positions[i]) for i in range(N)])
    
    # Track best solution
    best_idx = np.argmin(fitness)
    s.best = fitness[best_idx]
    s.bestIndividual = positions[best_idx].copy()
    s.convergence = []
    
    # Main loop
    for t in range(Max_time):
        
        # Calculate G(t) - decreasing gravitational constant
        G = G0 * np.exp(-alpha * t / Max_time)
        
        # Calculate Kbest - number of best agents that apply force (decreases over time)
        Kbest = max(1, int(N - (N - 1) * (t / Max_time)))
        
        # Calculate masses
        best_fit = np.min(fitness)
        worst_fit = np.max(fitness)
        
        if worst_fit != best_fit:
            # Normalize fitness to mass (better fitness = higher mass)
            masses = (fitness - worst_fit) / (best_fit - worst_fit)
        else:
            masses = np.ones(N)
        
        # Normalize masses
        masses = masses / np.sum(masses)
        
        # Get indices of Kbest agents (sorted by fitness)
        kbest_indices = np.argsort(fitness)[:Kbest]
        
        # Calculate forces and accelerations
        accelerations = np.zeros((N, dim))
        
        for i in range(N):
            force = np.zeros(dim)
            # Only Kbest agents contribute to the force
            for j in kbest_indices:
                if i != j:
                    # Euclidean distance
                    dist = np.linalg.norm(positions[i] - positions[j]) + 1e-10
                    # Gravitational force: F = G * m_j * (x_j - x_i) / dist
                    force += rng.random() * masses[j] * (positions[j] - positions[i]) / dist
            
            # Acceleration: a_i = G(t) * F_i
            accelerations[i] = G * force
        
        # Update velocities and positions
        velocities = rng.random((N, dim)) * velocities + accelerations
        positions = positions + velocities
        
        # Apply bounds
        positions = np.clip(positions, lb, ub)
        
        # Evaluate new positions
        fitness = np.array([objf(positions[i]) for i in range(N)])
        
        # Update best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < s.best:
            s.best = fitness[current_best_idx]
            s.bestIndividual = positions[current_best_idx].copy()
        
        # Store convergence
        s.convergence.append(s.best)
    
    s.endTime = time.time()
    s.executionTime = s.endTime - s.startTime
    s.convergence = np.array(s.convergence)
    
    return s
