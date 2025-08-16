# Hierarchical Visual Working Memory Model
This repository contains the simulation code for the hierarchical model of visual working memory (VWM) presented in the paper [PAPER TITLE]. The model investigates the structural, dynamic, and capacity properties of a biologically inspired VWM system with multiple feedback levels.
## Model Overview
The model consists of four main theoretical components:
1. **Structural Complexity**: Models the neuron count in a hierarchical VWM system with feedback, showing linear scaling in the number of retinal locations under stability conditions.
2. **Dynamic Convergence**: Simulates the neural dynamics of a single layer, demonstrating exponential convergence under gain constraints.
3. **Memory Capacity**: Computes the number of resolvable memories based on signal-to-noise ratio (SNR) considerations, incorporating hierarchical clarity and cross-talk.
4. **Task-Driven Retrieval**: Estimates the computational cost of retrieving complex events in hierarchical representations.
Additionally, a comprehensive parameter sweep analysis explores the memory capacity landscape across five key parameters.
## Repository Structure
- `test_complexity_theorem.py`: Main script to run the simulations for the four theorems and generate the integrated validation figure.
- `vwm_parameter_sweepp.py`: Script for the parameter sweep analysis of memory capacity.
- `hierarchical_test.py`: A simple test run
- `README.md`: This file.
## Dependencies
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- tqdm (for progress bars in parameter sweep)
Install dependencies via:
```bash
pip install numpy scipy matplotlib tqdm
```
## Results
The main simulation outputs a four-panel figure:
1. **Structural Complexity**: Linear scaling of total neuron count with retinal input size.
2. **Dynamic Convergence**: Neural activity convergence (stable) vs. oscillations (unstable).
3. **Memory Capacity**: Sublinear scaling of resolvable memories with available slots.
4. **Task Retrieval**: Linear scaling of retrieval cost with feature complexity.
The parameter sweep generates:
1. A four-panel figure showing the capacity landscape across parameter combinations.
2. Statistical analysis of parameter impacts on capacity.
