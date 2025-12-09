# -*- coding: utf-8 -*-
"""
Improved Hybrid MVO-PSO with Diversity Preservation
Addresses premature convergence through multiple mechanisms
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
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
    """Roulette wheel selection"""
    accumulation = np.cumsum(weights)
    p = np.random.random() * accumulation[-1]
    
    for index in range(len(accumulation)):
        if accumulation[index] > p:
            return index
    
    return len(weights) - 1


def calculate_diversity(population):
    """Calculate population diversity (average distance from centroid)"""
    centroid = np.mean(population, axis=0)
    distances = np.linalg.norm(population - centroid, axis=1)
    return np.mean(distances)


def levy_flight(dim, beta=1.5):
    """Generate Levy flight step for better exploration"""
    # Use Python's math module; numpy does not expose math.gamma
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v)**(1 / beta)
    
    return step


def sequential_hybrid(objf, lb, ub, dim, N, Max_time, 
                      switch_iter=None,
                      diversity_threshold=0.01,
                      mutation_rate=0.1,
                      elite_size=3):
    """
    Improved Hybrid MVO-PSO with Diversity Preservation
    
    Improvements to prevent premature convergence:
    1. Diversity monitoring and adaptive restart
    2. Levy flight for exploration
    3. Mutation operator
    4. Elite preservation
    5. Adaptive parameter control
    6. Opposition-based learning
    
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
        Maximum iterations
    switch_iter : int, optional
        Iteration to switch from MVO to PSO (default: 60% of Max_time)
    diversity_threshold : float
        Minimum diversity before triggering restart (default: 0.01)
    mutation_rate : float
        Probability of mutation (default: 0.1)
    elite_size : int
        Number of elite solutions to preserve (default: 3)
    
    Returns:
    --------
    s : Solution
        Solution object with results
    """
    
    # Set switch point (later than default for more exploration)
    if switch_iter is None:
        switch_iter = int(Max_time * 0.6)
    
    # MVO Parameters (standard values)
    WEP_Max = 1.0
    WEP_Min = 0.2
    p = 6
    
    # PSO Parameters (standard values)
    Vmax = 6
    wMax = 0.9
    wMin = 0.4
    c1 = 2.0  # Cognitive coefficient (standard)
    c2 = 2.0  # Social coefficient (standard)
    
    # Handle bounds
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize population with opposition-based learning
    Population = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb
    
    # Generate opposite population
    opposite_pop = lb + ub - Population
    
    # Evaluate both and select best N
    combined_pop = np.vstack([Population, opposite_pop])
    combined_fitness = np.array([objf(ind) for ind in combined_pop])
    best_indices = np.argsort(combined_fitness)[:N]
    Population = combined_pop[best_indices]
    
    # Initialize velocities
    vel = np.zeros((N, dim))
    
    # Initialize best solutions
    Best_position = np.zeros(dim)
    Best_fitness = float("inf")
    
    # Elite archive (store top solutions across iterations)
    elite_positions = []
    elite_fitness = []
    
    # Personal best for PSO
    pBest = Population.copy()
    pBestScore = np.full(N, float("inf"))
    
    convergence = np.zeros(Max_time)
    diversity_history = []
    restart_count = 0
    
    s = Solution()
    
    print(f'Improved Hybrid MVO-PSO is optimizing "{objf.__name__}"')
    print(f"Phase 1 (MVO): Iterations 1-{switch_iter}")
    print(f"Phase 2 (PSO): Iterations {switch_iter+1}-{Max_time}")
    print(f"Diversity threshold: {diversity_threshold}")
    
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for Time in range(1, Max_time + 1):
        
        # Evaluate fitness for all individuals
        Fitness = np.zeros(N)
        
        for i in range(N):
            Population[i, :] = np.clip(Population[i, :], lb, ub)
            Fitness[i] = objf(Population[i, :])
            
            if Fitness[i] < pBestScore[i]:
                pBestScore[i] = Fitness[i]
                pBest[i, :] = Population[i, :].copy()
            
            if Fitness[i] < Best_fitness:
                Best_fitness = Fitness[i]
                Best_position = Population[i, :].copy()
        
        # Update elite archive (keep best unique up to elite_size)
        all_positions = np.vstack([Population, np.array(elite_positions)]) if elite_positions else Population
        all_fitness = np.concatenate([Fitness, elite_fitness]) if elite_fitness else Fitness
        elite_indices = np.argsort(all_fitness)
        # Build top list up to elite_size
        new_elites_pos = []
        new_elites_fit = []
        for idx in elite_indices:
            cand_pos = all_positions[idx]
            cand_fit = all_fitness[idx]
            # Avoid near-duplicates (by position)
            is_dup = any(np.allclose(cand_pos, ep) for ep in new_elites_pos)
            if not is_dup:
                new_elites_pos.append(cand_pos)
                new_elites_fit.append(cand_fit)
            if len(new_elites_pos) >= elite_size:
                break
        elite_positions = new_elites_pos
        elite_fitness = new_elites_fit
        
        # Calculate and monitor diversity
        diversity = calculate_diversity(Population)
        diversity_history.append(diversity)
        
        # Check for premature convergence and restart if needed
        if diversity < diversity_threshold and Time < Max_time * 0.8:
            print(f"  → Diversity too low ({diversity:.6f}) at iteration {Time}. Restarting population...")
            restart_count += 1
            
            # Keep elite solutions
            for i in range(min(elite_size, N)):
                Population[i] = elite_positions[i].copy()
            
            # Reinitialize rest with Levy flight around best
            for i in range(elite_size, N):
                if np.random.random() < 0.5:
                    # Levy flight around best position
                    levy_step = levy_flight(dim)
                    Population[i] = Best_position + levy_step * (ub - lb) * 0.1
                else:
                    # Random initialization
                    Population[i] = np.random.uniform(0, 1, dim) * (ub - lb) + lb
                
                Population[i] = np.clip(Population[i], lb, ub)
            
            # Reset velocities
            vel = np.zeros((N, dim))
        
        # ============ PHASE 1: MVO ============
        if Time <= switch_iter:
            # WEP and TDR scaled to MVO phase duration
            mvo_progress = Time / switch_iter
            WEP = WEP_Min + mvo_progress * (WEP_Max - WEP_Min)
            TDR = 1 - (Time ** (1/p)) / (switch_iter ** (1/p))
            
            # Sort universes
            sorted_indexes = np.argsort(Fitness)
            Sorted_population = Population[sorted_indexes, :]
            sorted_Fitness = Fitness[sorted_indexes]
            normalized_sorted_Fitness = normr(sorted_Fitness)
            
            # Keep best
            Population[0, :] = Sorted_population[0, :]
            
            # Update using MVO with Levy flight
            for i in range(1, N):
                for j in range(dim):
                    r1 = np.random.random()
                    
                    if r1 < normalized_sorted_Fitness[i]:
                        White_hole_index = roulette_wheel_selection(-sorted_Fitness)
                        Population[i, j] = Sorted_population[White_hole_index, j]
                    
                    r2 = np.random.random()
                    if r2 < WEP:
                        r3 = np.random.random()
                        r4 = np.random.random()
                        
                        if r3 < 0.5:
                            Population[i, j] = (Best_position[j] + 
                                              TDR * ((ub[j] - lb[j]) * r4 + lb[j]))
                        else:
                            Population[i, j] = (Best_position[j] - 
                                              TDR * ((ub[j] - lb[j]) * r4 + lb[j]))
                    
                    # Levy flight mutation
                    if np.random.random() < mutation_rate:
                        levy_step = levy_flight(1)[0]
                        Population[i, j] += levy_step * (ub[j] - lb[j]) * 0.01
        
        # ============ PHASE 2: PSO with improvements ============
        else:
            # Adaptive inertia weight (nonlinear)
            pso_iter = Time - switch_iter
            pso_max_iter = Max_time - switch_iter
            progress = pso_iter / pso_max_iter
            w = wMax - (wMax - wMin) * progress**2  # Quadratic decay
            
            # Update velocities and positions
            for i in range(N):
                for j in range(dim):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    # Standard PSO update
                    vel[i, j] = (w * vel[i, j] +
                                c1 * r1 * (pBest[i, j] - Population[i, j]) +
                                c2 * r2 * (Best_position[j] - Population[i, j]))
                    
                    # Velocity clamping
                    vel[i, j] = np.clip(vel[i, j], -Vmax, Vmax)
                    
                    # Position update
                    Population[i, j] = Population[i, j] + vel[i, j]
                    
                    # Mutation for diversity
                    if np.random.random() < mutation_rate * (1 - progress):
                        Population[i, j] += np.random.randn() * (ub[j] - lb[j]) * 0.01
        
        # Store convergence
        convergence[Time - 1] = Best_fitness
        
        # Print progress
        if Time == 1 or Time == switch_iter or Time == switch_iter + 1 or Time % 20 == 0 or Time == Max_time:
            phase = "MVO" if Time <= switch_iter else "PSO"
            print(f"[{phase}] Iter {Time}/{Max_time}: Best={Best_fitness:.6e}, Diversity={diversity:.6f}")
    
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = f"Sequential_Hybrid (restarts: {restart_count})"
    s.bestIndividual = Best_position
    s.best = Best_fitness
    s.objfname = objf.__name__
    s.lb = lb
    s.ub = ub
    s.dim = dim
    s.popnum = N
    s.maxiters = Max_time
    
    print(f"\nTotal diversity-based restarts: {restart_count}")
    print(f"Final diversity: {diversity_history[-1]:.6f}")
    # Print best solution
    print(f"Best solution found → {Best_fitness:.6e}: {Best_position}")
    
    return s


# Test functions
def sphere(x):
    return np.sum(x**2)


def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def f8(x):
    return np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


if __name__ == "__main__":
    print("="*70)
    print("Testing Hybrid MVO-PSO")
    print("="*70)
    
    # Run optimization
    result = sequential_hybrid(sphere, lb=-75, ub=140, dim=3, N=30, Max_time=100)
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Best fitness: {result.best:.6e}")
    print(f"Execution time: {result.executionTime:.2f} seconds")
    print(f"Optimizer: {result.optimizer}")
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Main convergence curve
    ax1 = fig.add_subplot(gs[0, :])
    iterations = np.arange(1, len(result.convergence) + 1)
    switch_point = int(result.maxiters * 0.6)
    
    ax1.plot(iterations, result.convergence, linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax1.axvline(x=switch_point, color='#E63946', linestyle='--', 
                linewidth=2, label=f'Switch Point (iter {switch_point})', alpha=0.8)
    ax1.fill_between(iterations[:switch_point], 0, result.convergence[:switch_point], 
                     alpha=0.2, color='green', label='MVO Phase')
    ax1.fill_between(iterations[switch_point:], 0, result.convergence[switch_point:], 
                     alpha=0.2, color='orange', label='PSO Phase')
    
    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontsize=13, fontweight='bold')
    ax1.set_title(f'Hybrid MVO-PSO Convergence - {result.objfname}', 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='upper right')
    
    # Plot 2: Log scale convergence
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(iterations, result.convergence, linewidth=2, color='#A23B72', marker='o', 
                 markersize=3, markevery=max(1, len(iterations)//20))
    ax2.axvline(x=switch_point, color='#E63946', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Best Fitness (log scale)', fontsize=11)
    ax2.set_title('Convergence (Log Scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle=':')
    
    # Plot 3: Phase comparison
    ax3 = fig.add_subplot(gs[1, 1])
    mvo_iters = np.arange(1, switch_point + 1)
    pso_iters = np.arange(switch_point + 1, len(result.convergence) + 1)
    
    ax3.plot(mvo_iters, result.convergence[:switch_point], linewidth=2.5, 
             color='#06A77D', label='MVO Phase', marker='s', markersize=4, 
             markevery=max(1, len(mvo_iters)//10))
    ax3.plot(pso_iters, result.convergence[switch_point:], linewidth=2.5, 
             color='#F77F00', label='PSO Phase', marker='^', markersize=4,
             markevery=max(1, len(pso_iters)//10))
    
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Best Fitness', fontsize=11)
    ax3.set_title('Phase-wise Performance', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10)
    
    # Plot 4: Improvement rate
    ax4 = fig.add_subplot(gs[2, 0])
    improvement = np.abs(np.diff(result.convergence))
    improvement = np.concatenate([[0], improvement])  # Add 0 for first iteration
    
    ax4.plot(iterations, improvement, linewidth=1.5, color='#D62828', alpha=0.7)
    ax4.axvline(x=switch_point, color='#E63946', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.fill_between(iterations, 0, improvement, alpha=0.3, color='#D62828')
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Improvement Rate', fontsize=11)
    ax4.set_title('Fitness Improvement per Iteration', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 5: Statistics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"""
    Algorithm Statistics
    {'='*40}
    
    Function: {result.objfname}
    Dimensions: {result.dim}
    Population Size: {result.popnum}
    Total Iterations: {result.maxiters}
    
    Best Fitness: {result.best:.6e}
    Initial Fitness: {result.convergence[0]:.6e}
    Total Improvement: {result.convergence[0] - result.best:.6e}
    
    MVO Phase (1-{switch_point}):
      Start: {result.convergence[0]:.6e}
      End: {result.convergence[switch_point-1]:.6e}
      Improvement: {result.convergence[0] - result.convergence[switch_point-1]:.6e}
    
    PSO Phase ({switch_point+1}-{result.maxiters}):
      Start: {result.convergence[switch_point]:.6e}
      End: {result.convergence[-1]:.6e}
      Improvement: {result.convergence[switch_point] - result.convergence[-1]:.6e}
    
    Execution Time: {result.executionTime:.2f} seconds
    """
    
    ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Hybrid MVO-PSO Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.show()
