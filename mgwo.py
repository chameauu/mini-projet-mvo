"""
Modified Grey Wolf Optimizer (MGWO)
Based on the social hierarchy of grey wolves (alpha, beta, delta, omega)

The algorithm mimics the leadership hierarchy and hunting mechanism of grey wolves:
- Alpha (α): Best solution (leader)
- Beta (β): Second best solution
- Delta (δ): Third best solution
- Omega (ω): Rest of the population

Enhanced with Cauchy and Gaussian perturbations for better exploration.
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


def MGWO(objf, lb, ub, dim, N, Max_time):
    """
    Modified Grey Wolf Optimizer
    
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
        Population size (number of wolves)
    Max_time : int
        Maximum iterations
    
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
    
    # Initialize wolf population
    X = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb
    
    # Initialize alpha, beta, and delta positions
    X_alpha = np.zeros(dim)
    X_beta = np.zeros(dim)
    X_delta = np.zeros(dim)
    
    # Initialize fitness
    fitness_alpha = float("inf")
    fitness_beta = float("inf")
    fitness_delta = float("inf")
    
    convergence = np.zeros(Max_time)
    
    s = Solution()
    
    print(f'MGWO is optimizing "{objf.__name__}"')
    
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for t in range(Max_time):
        
        # Evaluate fitness for all wolves
        fitness = np.zeros(N)
        
        for i in range(N):
            # Apply bounds
            X[i, :] = np.clip(X[i, :], lb, ub)
            
            # Evaluate fitness
            fitness[i] = objf(X[i, :])
            
            # Update alpha, beta, and delta
            if fitness[i] < fitness_alpha:
                fitness_delta = fitness_beta
                X_delta = X_beta.copy()
                
                fitness_beta = fitness_alpha
                X_beta = X_alpha.copy()
                
                fitness_alpha = fitness[i]
                X_alpha = X[i, :].copy()
            
            elif fitness[i] < fitness_beta:
                fitness_delta = fitness_beta
                X_delta = X_beta.copy()
                
                fitness_beta = fitness[i]
                X_beta = X[i, :].copy()
            
            elif fitness[i] < fitness_delta:
                fitness_delta = fitness[i]
                X_delta = X[i, :].copy()
        
        # Linearly decreasing parameter 'a' from 2 to 0
        a = 2 - 2 * (t / Max_time)
        
        # Update position of each wolf
        for i in range(N):
            
            # ========== Update based on alpha wolf ==========
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            
            D_alpha = np.abs(C1 * X_alpha - X[i, :])
            X1 = X_alpha - A1 * D_alpha
            
            # ========== Update based on beta wolf ==========
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            
            D_beta = np.abs(C2 * X_beta - X[i, :])
            X2 = X_beta - A2 * D_beta
            
            # ========== Update based on delta wolf ==========
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            
            D_delta = np.abs(C3 * X_delta - X[i, :])
            X3 = X_delta - A3 * D_delta
            
            # ========== Calculate new position (average of three) ==========
            X_new = (X1 + X2 + X3) / 3.0
            
            # ========== Enhancement: Cauchy + Gaussian perturbation ==========
            # Gamma decreases over iterations (more exploration early, exploitation later)
            gamma = 1 - (t / Max_time) ** 2
            
            # Cauchy perturbation (heavy-tailed, good for exploration)
            cauchy_step = gamma * np.random.standard_cauchy(dim)
            
            # Gaussian perturbation (light-tailed, good for exploitation)
            gauss_step = (1 - gamma) * np.random.normal(0, 1, dim)
            
            # Combined perturbation
            perturbation = cauchy_step + gauss_step
            
            # Apply perturbation
            X_new = X_new + perturbation
            
            # Apply bounds
            X_new = np.clip(X_new, lb, ub)
            
            # Update wolf position
            X[i, :] = X_new
        
        # Store convergence
        convergence[t] = fitness_alpha
        
        # Print progress
        if t % 10 == 0 or t == 0 or t == Max_time - 1:
            print(f"Iteration {t + 1}/{Max_time}: Best fitness = {fitness_alpha:.6e}")
    
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "MGWO"
    s.bestIndividual = X_alpha
    s.best = fitness_alpha
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


def schwefel(x):
    """Schwefel function"""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock(x):
    """Rosenbrock function"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley(x):
    """Ackley function"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing Modified Grey Wolf Optimizer (MGWO)")
    print("="*60)
    
    # Test on Sphere function
    print("\nTest 1: Sphere Function (Unimodal)")
    print("-"*60)
    result = GWO(sphere, lb=-100, ub=100, dim=30, N=30, Max_time=200)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Best fitness: {result.best:.6e}")
    print(f"Execution time: {result.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result.bestIndividual[:5]}")
    
    # Test on Rastrigin function
    print("\n\n" + "="*60)
    print("Test 2: Rastrigin Function (Multimodal)")
    print("-"*60)
    result2 = GWO(rastrigin, lb=-5.12, ub=5.12, dim=30, N=30, Max_time=200)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Best fitness: {result2.best:.6e}")
    print(f"Execution time: {result2.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result2.bestIndividual[:5]}")
    
    # Visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Sphere convergence
        axes[0].plot(result.convergence, linewidth=2, color='#2E86AB')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Best Fitness', fontsize=12)
        axes[0].set_title(f'MGWO on {result.objfname}', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Plot 2: Rastrigin convergence
        axes[1].plot(result2.convergence, linewidth=2, color='#E63946')
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Best Fitness', fontsize=12)
        axes[1].set_title(f'MGWO on {result2.objfname}', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nSkipping visualization: {e}")
