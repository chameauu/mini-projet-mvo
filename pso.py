# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
"""

import random
import numpy
import time
import matplotlib.pyplot as plt


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0


def PSO(objf, lb, ub, dim, PopSize, iters):

    # PSO parameters

    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    vel = numpy.zeros((PopSize, dim))

    pBestScore = numpy.zeros(PopSize)
    pBestScore.fill(float("inf"))

    pBest = numpy.zeros((PopSize, dim))
    gBest = numpy.zeros(dim)

    gBestScore = float("inf")

    pos = numpy.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    convergence_curve = numpy.zeros(iters)

    ############################################
    print('PSO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        for i in range(0, PopSize):
            # pos[i,:]=checkBounds(pos[i,:],lb,ub)
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])
            # Calculate objective function for each particle
            fitness = objf(pos[i, :])

            if pBestScore[i] > fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax

                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore

        # Print progress every 10 iterations (same as MVO for fair comparison)
        if l % 10 == 0 or l == 0 or l == iters - 1:
            print(f"Iteration {l + 1}/{iters}: Best fitness = {gBestScore:.6e}")
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.bestIndividual = gBest
    s.objfname = objf.__name__

    return s


# Test functions
def sphere(x):
    """Sphere function - f(0) = 0"""
    return numpy.sum(x**2)


def rastrigin(x):
    """Rastrigin function - f(0) = 0"""
    n = len(x)
    return 10 * n + numpy.sum(x**2 - 10 * numpy.cos(2 * numpy.pi * x))


def f8(x):
    """Schwefel function"""
    return numpy.sum(x * numpy.sin(numpy.sqrt(numpy.abs(x))))


# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing PSO on Schwefel Function (f8)")
    print("="*60)
    
    result = PSO(sphere, lb=-75, ub=140, dim=30, PopSize=30, iters=100)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Best fitness: {result.convergence[-1]:.6e}")
    print(f"Execution time: {result.executionTime:.2f} seconds")
    print(f"Best solution (first 5 dims): {result.bestIndividual[:5]}")
    
    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(result.convergence, linewidth=2, color='blue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(f'PSO Convergence Curve - {result.objfname}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.show()
