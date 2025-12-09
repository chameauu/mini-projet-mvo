import numpy as np
import time
from sklearn.preprocessing import normalize
from cec2017.functions import f1


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
    
    return len(weights) - 1  # Return last index if nothing selected


def MVO(objf, lb, ub, dim, N, Max_time):
    """
    Multi-Verse Optimizer
    
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
        Number of universes (population size)
    Max_time : int
        Maximum iterations
    
    Returns:
    --------
    s : Solution
        Solution object with results
    """
    
    # Parameters
    WEP_Max = 1.0
    WEP_Min = 0.2
    p = 6  # For TDR calculation
    
    # Handle bounds
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize universes
    Universes = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb
    
    # Initialize best
    Best_universe = np.zeros(dim)
    Best_universe_Inflation_rate = float("inf")
    
    convergence = np.zeros(Max_time)
    
    s = Solution()
    
    print(f'MVO is optimizing "{objf.__name__}"')
    
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for Time in range(1, Max_time + 1):
        
        # Update WEP (Wormhole Existence Probability)
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_time)
        
        # Update TDR (Traveling Distance Rate)
        TDR = 1 - (Time ** (1/p)) / (Max_time ** (1/p))
        
        # Evaluate fitness (inflation rates)
        Inflation_rates = np.zeros(N)
        
        for i in range(N):
            # Apply bounds
            Universes[i, :] = np.clip(Universes[i, :], lb, ub)
            
            # Evaluate
            Inflation_rates[i] = objf(Universes[i, :])
            
            # Update best
            if Inflation_rates[i] < Best_universe_Inflation_rate:
                Best_universe_Inflation_rate = Inflation_rates[i]
                Best_universe = Universes[i, :].copy()
        
        # Sort universes by inflation rate
        sorted_indexes = np.argsort(Inflation_rates)
        Sorted_universes = Universes[sorted_indexes, :]
        sorted_Inflation_rates = Inflation_rates[sorted_indexes]
        
        # Normalize inflation rates for roulette wheel
        normalized_sorted_Inflation_rates = normr(sorted_Inflation_rates)
        
        # Keep best universe
        Universes[0, :] = Sorted_universes[0, :]
        
        # Update other universes
        for i in range(1, N):
            for j in range(dim):
                r1 = np.random.random()
                
                # White hole / Black hole mechanism
                if r1 < normalized_sorted_Inflation_rates[i]:
                    # Select white hole using roulette wheel
                    White_hole_index = roulette_wheel_selection(
                        -sorted_Inflation_rates  # Negative for minimization
                    )
                    
                    # Exchange object
                    Universes[i, j] = Sorted_universes[White_hole_index, j]
                
                # Wormhole mechanism
                r2 = np.random.random()
                if r2 < WEP:
                    r3 = np.random.random()
                    r4 = np.random.random()
                    
                    if r3 < 0.5:
                        # Move toward best universe
                        Universes[i, j] = (Best_universe[j] + 
                                          TDR * ((ub[j] - lb[j]) * r4 + lb[j]))
                    else:
                        # Move away from best universe
                        Universes[i, j] = (Best_universe[j] - 
                                          TDR * ((ub[j] - lb[j]) * r4 + lb[j]))
        
        # Store convergence
        convergence[Time - 1] = Best_universe_Inflation_rate
        
        # Print progress
        if Time % 10 == 0 or Time == 1:
            print(f"Iteration {Time}/{Max_time}: Best fitness = {Best_universe_Inflation_rate:.6e}")
    
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "MVO"
    s.bestIndividual = Best_universe
    s.best = Best_universe_Inflation_rate
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

def f8(x):
    return np.sum(x * np.sin(np.sqrt(np.abs(x))))



# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing MVO on Sphere Function")
    print("="*60)
    
    result = MVO(f1, lb=-75, ub=140, dim=10, N=30, Max_time=100)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Best fitness: {result.best:.6e}")
    print(f"Execution time: {result.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result.bestIndividual[:5]}")

    # Visualization: Convergence curve (display only, do not save)
    try:
        import matplotlib.pyplot as plt

        iters = np.arange(1, result.maxiters + 1) if hasattr(result, "maxiters") else np.arange(1, len(result.convergence) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(iters, result.convergence, label="Best fitness", color="tab:blue")
        plt.xlabel("Iteration")
        plt.ylabel("Best fitness")
        plt.title(f"MVO Convergence on {result.objfname}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Show the plot (may be skipped in some environments)
        try:
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print(f"Skipping plot due to: {e}")