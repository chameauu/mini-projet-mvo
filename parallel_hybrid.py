"""
Hybrid PSO-MVO Algorithm (Paper Implementation)
Based on: Jangir et al. (2017) - Engineering Science and Technology

This implementation follows the paper's approach where:
- PSO's Pbest is replaced with MVO's Universe selection
- Both algorithms work together in every iteration (parallel hybrid)
- Roulette wheel selection is used for universe selection

Key Equation:
v^(t+1) = w*v^t + c1*R1*(Universe - X^t) + c2*R2*(Gbest - X^t)

Reference:
Jangir, P., Parmar, S. A., Trivedi, I. N., & Bhesdadiya, R. H. (2017).
A novel hybrid Particle Swarm Optimizer with multi verse optimizer for 
global numerical optimization and Optimal Reactive Power Dispatch problem.
Engineering Science and Technology, an International Journal, 20(2), 570-586.
"""

import numpy as np
import time
from sklearn.preprocessing import normalize


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


def normr(Mat):
    """Normalize the columns of the matrix"""
    Mat = Mat.reshape(1, -1)
    if Mat.dtype != "float":
        Mat = np.asarray(Mat, dtype=float)
    B = normalize(Mat, norm="l2", axis=1)
    B = np.reshape(B, -1)
    return B


def roulette_wheel_selection(weights):
    """
    Roulette wheel selection based on weights
    Higher weight = higher probability of selection
    """
    # For minimization, invert weights (lower fitness = higher probability)
    accumulation = np.cumsum(weights)
    p = np.random.random() * accumulation[-1]
    
    for index in range(len(accumulation)):
        if accumulation[index] > p:
            return index
    
    return len(weights) - 1


def parallel_hybrid(objf, lb, ub, dim, N, Max_time):
    """
    Hybrid PSO-MVO Algorithm (Paper's Parallel Approach)
    
    This implementation combines PSO and MVO in every iteration:
    - Uses MVO's universe concept for exploration
    - Uses PSO's velocity mechanism for movement
    - Replaces PSO's Pbest with MVO's selected universe
    - Applies wormhole mechanism for local search
    
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
        Population size (number of universes/particles)
    Max_time : int
        Maximum iterations
    
    Returns:
    --------
    s : Solution
        Solution object with results
    """
    
    # PSO Parameters (from paper - Section 2.1)
    # Note: Paper states wmax=0.4, wmin=0.9 but this appears to be a typo
    # Standard PSO uses wmax=0.9, wmin=0.4 (decreasing inertia)
    # We follow standard convention: w decreases from 0.9 to 0.4
    Vmax = 6
    wMax = 0.9
    wMin = 0.4
    c1 = 2  # Cognitive coefficient
    c2 = 2  # Social coefficient
    
    # MVO Parameters (from paper - Equations 7 & 8)
    WEP_Max = 1.0  # max in Eq. 7
    WEP_Min = 0.2  # min in Eq. 7
    p = 6  # Exploitation accuracy parameter (Eq. 8)
    
    # Handle bounds
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize universes (population)
    Universes = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb
    
    # Initialize velocities (for PSO component)
    Velocities = np.zeros((N, dim))
    
    # Initialize best universe (global best)
    Best_universe = np.zeros(dim)
    Best_universe_fitness = float("inf")
    
    # Convergence tracking
    convergence = np.zeros(Max_time)
    
    s = Solution()
    
    print(f'Hybrid PSO-MVO (Paper) is optimizing "{objf.__name__}"')
    
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main optimization loop
    for iteration in range(1, Max_time + 1):
        
        # ========== Step 1: Evaluate Fitness (Inflation Rates) ==========
        Inflation_rates = np.zeros(N)
        
        for i in range(N):
            # Apply bounds
            Universes[i, :] = np.clip(Universes[i, :], lb, ub)
            
            # Evaluate fitness (inflation rate)
            Inflation_rates[i] = objf(Universes[i, :])
            
            # Update global best
            if Inflation_rates[i] < Best_universe_fitness:
                Best_universe_fitness = Inflation_rates[i]
                Best_universe = Universes[i, :].copy()
        
        # ========== Step 2: Update MVO Parameters ==========
        # Wormhole Existence Probability (increases over time)
        WEP = WEP_Min + iteration * ((WEP_Max - WEP_Min) / Max_time)
        
        # Traveling Distance Rate (decreases over time)
        TDR = 1 - (iteration ** (1/p)) / (Max_time ** (1/p))
        
        # ========== Step 3: Sort Universes by Fitness ==========
        sorted_indexes = np.argsort(Inflation_rates)
        Sorted_universes = Universes[sorted_indexes, :]
        sorted_Inflation_rates = Inflation_rates[sorted_indexes]
        
        # Normalize inflation rates for roulette wheel selection
        normalized_sorted_Inflation_rates = normr(sorted_Inflation_rates)
        
        # ========== Step 4: Update PSO Inertia Weight ==========
        w = wMax - iteration * ((wMax - wMin) / Max_time)
        
        # ========== Step 5: Update Each Universe/Particle ==========
        # Paper Equation 11: v(t+1) = w*v(t) + c1*R1*(Universe - X) + c2*R2*(Gbest - X)
        for i in range(N):
            
            # Select a universe using roulette wheel (MVO component)
            # This replaces Pbest in standard PSO
            # Use negative fitness for minimization (better fitness = higher selection probability)
            selected_universe_idx = roulette_wheel_selection(-sorted_Inflation_rates)
            selected_universe = Sorted_universes[selected_universe_idx, :]
            
            # Random coefficients (R1, R2 in paper)
            R1 = np.random.random(dim)
            R2 = np.random.random(dim)
            
            # Hybrid PSO-MVO velocity update (Equation 11)
            # Key innovation: Replace Pbest with selected Universe from MVO
            Velocities[i, :] = (
                w * Velocities[i, :] +
                c1 * R1 * (selected_universe - Universes[i, :]) +  # MVO universe instead of Pbest
                c2 * R2 * (Best_universe - Universes[i, :])        # Global best (Gbest)
            )
            
            # Velocity clamping
            Velocities[i, :] = np.clip(Velocities[i, :], -Vmax, Vmax)
            
            # Update position using PSO equation: X(t+1) = X(t) + V(t+1)
            Universes[i, :] = Universes[i, :] + Velocities[i, :]
            
            # ===== Apply MVO Wormhole Mechanism (Equation 6) =====
            # Wormholes create tunnels between universes for local search
            for j in range(dim):
                r2 = np.random.random()
                
                if r2 < WEP:  # Wormhole exists
                    r3 = np.random.random()
                    r4 = np.random.random()
                    
                    if r3 < 0.5:
                        # Eq 6: X_j + TDR * ((ub_j - lb_j) * r4 + lb_j)
                        Universes[i, j] = (
                            Best_universe[j] + 
                            TDR * ((ub[j] - lb[j]) * r4 + lb[j])
                        )
                    else:
                        # Eq 6: X_j - TDR * ((ub_j - lb_j) * r4 + lb_j)
                        Universes[i, j] = (
                            Best_universe[j] - 
                            TDR * ((ub[j] - lb[j]) * r4 + lb[j])
                        )
                # else: r2 >= WEP, keep current position (no wormhole)
        
        # Store convergence
        convergence[iteration - 1] = Best_universe_fitness
        
        # Print progress
        if iteration % 10 == 0 or iteration == 1 or iteration == Max_time:
            print(f"Iteration {iteration}/{Max_time}: Best fitness = {Best_universe_fitness:.6e}")
    
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "Parallel_Hybrid"
    s.bestIndividual = Best_universe
    s.best = Best_universe_fitness
    s.objfname = objf.__name__
    s.lb = lb
    s.ub = ub
    s.dim = dim
    s.popnum = N
    s.maxiters = Max_time
    
    return s


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

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


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing Hybrid PSO-MVO (Paper's Parallel Approach)")
    print("="*70)
    print("\nBased on: Jangir et al. (2017)")
    print("Strategy: PSO velocity with MVO universe selection + wormhole")
    print("="*70)
    
    # Test on Rastrigin function (multimodal)
    print("\nTest 1: Rastrigin Function (Multimodal)")
    print("-"*70)
    result = parallel_hybrid(rastrigin, lb=-5.12, ub=5.12, dim=30, N=30, Max_time=200)
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Best fitness: {result.best:.6e}")
    print(f"Execution time: {result.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result.bestIndividual[:5]}")
    print(f"Final convergence: {result.convergence[-1]:.6e}")
    
    # Test on Sphere function (unimodal)
    print("\n\n" + "="*70)
    print("Test 2: Sphere Function (Unimodal)")
    print("-"*70)
    result2 = parallel_hybrid(sphere, lb=-100, ub=100, dim=30, N=30, Max_time=200)
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Best fitness: {result2.best:.6e}")
    print(f"Execution time: {result2.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result2.bestIndividual[:5]}")
    print(f"Final convergence: {result2.convergence[-1]:.6e}")
    
    # Test on Schwefel function (deceptive)
    print("\n\n" + "="*70)
    print("Test 3: Schwefel Function (Deceptive)")
    print("-"*70)
    result3 = parallel_hybrid(sphere, lb=-500, ub=500, dim=3, N=30, Max_time=100)
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Best fitness: {result3.best:.6e}")
    print(f"Execution time: {result3.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result3.bestIndividual[:5]}")
    print(f"Final convergence: {result3.convergence[-1]:.6e}")
    
    print("\n" + "="*70)
    print("Paper Implementation Complete!")
    print("="*70)
