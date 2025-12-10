# Multi-Verse Optimizer (MVO) and Hybrid Algorithms

A comprehensive implementation of the Multi-Verse Optimizer (MVO) and various hybrid metaheuristic optimization algorithms for solving complex optimization problems. This project includes implementations of several optimization algorithms and their hybrid variants, tested on the CEC2017 benchmark functions.

## Overview

This project contains implementations of the following optimization algorithms:

### Standalone Algorithms
- **MVO** (Multi-Verse Optimizer): Based on the physics of white holes, black holes, and wormholes
- **PSO** (Particle Swarm Optimization): Inspired by bird flocking and fish schooling behaviors
- **GA** (Genetic Algorithm): Based on evolutionary principles and natural selection
- **GSA** (Gravitational Search Algorithm): Based on Newton's law of gravity and motion
- **MGWO** (Modified Grey Wolf Optimizer): Based on the social hierarchy and hunting mechanism of grey wolves

### Hybrid Algorithms
- **Parallel Hybrid PSO-MVO**: Combines PSO and MVO in parallel, where PSO's personal best is replaced with MVO's universe selection
- **Sequential Hybrid MVO-PSO**: Improved hybrid with diversity preservation mechanisms to address premature convergence

## Features

- Implementation of multiple metaheuristic optimization algorithms
- Hybrid algorithms combining strengths of different optimizers
- Tested on CEC2017 benchmark functions (30 standard test functions)
- Support for multi-dimensional optimization problems
- Convergence tracking and performance visualization
- Jupyter notebook with comprehensive experiments and analysis

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chameauu/mini-projet-mvo.git
cd mini-projet-mvo
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the CEC2017 benchmark functions:
```bash
cd cec2017-py
python3 setup.py install
cd ..
```

## Dependencies

The project requires the following Python packages:
- `numpy>=1.20.0` - Numerical computations
- `scikit-learn>=0.24.0` - Normalization and preprocessing
- `matplotlib>=3.3.0` - Visualization and plotting
- `pandas>=1.2.0` - Data manipulation and analysis

## Usage

### Using Individual Algorithms

```python
from mvo import MVO
from cec2017.functions import f1

# Define problem parameters
dim = 30  # Dimension
lb = -100  # Lower bound
ub = 100   # Upper bound
N = 50     # Population size
Max_time = 1000  # Maximum iterations

# Run MVO
solution = MVO(f1, lb, ub, dim, N, Max_time)
print(f"Best fitness: {solution.best}")
print(f"Best solution: {solution.bestIndividual}")
```

### Using Hybrid Algorithms

```python
from parallel_hybrid import parallel_hybrid
from cec2017.functions import f1

# Run Parallel Hybrid PSO-MVO
solution = parallel_hybrid(f1, lb, ub, dim, N, Max_time)
print(f"Best fitness: {solution.best}")
```

### Running Experiments with Jupyter Notebook

Open the `cec2017_mvo_hybrids.ipynb` notebook to:
- Run comprehensive experiments on CEC2017 benchmark functions
- Compare performance of different algorithms
- Visualize convergence curves
- Analyze statistical results

```bash
jupyter notebook cec2017_mvo_hybrids.ipynb
```

## Project Structure

```
mini-projet-mvo/
├── mvo.py                      # Multi-Verse Optimizer implementation
├── pso.py                      # Particle Swarm Optimization implementation
├── ga.py                       # Genetic Algorithm implementation
├── gsa.py                      # Gravitational Search Algorithm implementation
├── mgwo.py                     # Modified Grey Wolf Optimizer implementation
├── parallel_hybrid.py          # Parallel Hybrid PSO-MVO implementation
├── sequential_hybrid.py        # Sequential Hybrid MVO-PSO implementation
├── cec2017_mvo_hybrids.ipynb  # Jupyter notebook with experiments
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── results/                    # Experimental results
│   └── results_incremental.csv
└── cec2017-py/                # CEC2017 benchmark functions
    ├── cec2017/               # CEC2017 module
    └── setup.py               # Installation script
```

## Algorithm Details

### Multi-Verse Optimizer (MVO)
MVO is inspired by the multiverse theory in physics. It uses concepts of white holes, black holes, and wormholes to explore and exploit the search space.

### Parallel Hybrid PSO-MVO
Based on the paper by Jangir et al. (2017), this hybrid algorithm combines:
- PSO's velocity update mechanism
- MVO's universe selection via roulette wheel
- Global best from both algorithms

Key equation: `v^(t+1) = w*v^t + c1*R1*(Universe - X^t) + c2*R2*(Gbest - X^t)`

### Sequential Hybrid MVO-PSO
An improved hybrid approach with:
- Diversity preservation mechanisms
- Adaptive parameter control
- Mechanisms to prevent premature convergence

## CEC2017 Benchmark Functions

The project uses the CEC2017 single objective optimization benchmark suite, which includes:
- 30 test functions (f1-f30)
- Support for dimensions: 2, 10, 20, 30, 50, 100
- Various problem types: unimodal, multimodal, hybrid, composition

## Results

Results from experiments are saved in the `results/` directory:
- `results_incremental.csv`: Performance metrics for all algorithms across benchmark functions

## References

1. Mirjalili, S., Mirjalili, S. M., & Hatamlou, A. (2016). Multi-verse optimizer: a nature-inspired algorithm for global optimization. *Neural Computing and Applications*, 27(2), 495-513.

2. Jangir, P., Parmar, S. A., Trivedi, I. N., & Bhesdadiya, R. H. (2017). A novel hybrid Particle Swarm Optimizer with multi verse optimizer for global numerical optimization and Optimal Reactive Power Dispatch problem. *Engineering Science and Technology, an International Journal*, 20(2), 570-586.

3. Awad, N. H., Ali, M. Z., Suganthan, P. N., Liang, J. J., & Qu, B. Y. (2016). Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Bound Constrained Real-Parameter Numerical Optimization.

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or suggestions, please open an issue in the repository.
