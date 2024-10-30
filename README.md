# SUNSpinsInCavityQED

Python code for simulation of SU(N) spins in cavity-QED system and reproduce results in https://arxiv.org/abs/2210.14224.
The Hamiltonian is detailed in Eq. (1-2) in https://arxiv.org/abs/2210.14224.
By representing the SU(N) spins via Schwinger bosons, it is possible to initialize two kind of initial states (see Sec. IIC of the paper for more details):
1) mean field states: bosonic coherent states on each Schwinger boson mode.
2) Gaussian states: multi-mode Gaussian state on each site.

## Content
src : contains python code for reproducing results in https://arxiv.org/abs/2210.14224

library : contains the methods needed

script_plot : contains script for creating figures 

scaling_analysis : supplemental information such as scaling analysis in the timestep and system size for checking convergence of results.

## Prerequisites

Standard Python libraries including numpy, Scipy.

## Usage
The scripts for reproducing the results in https://arxiv.org/abs/2210.14224 are in the src folder.

The main ones are SUN_mean_field.py and SUN_Gaussian_evolution.py (the other scripts are variation of these).

Both SUN_mean_field.py and SUN_Gaussian_evolution.py need inputs passed either via a json file, or via prompt using 
```bash
--name_variable=value_variable
```
The variables needed are explained in the input.py script contained in the folder library.

## Installation

Clone the repository:
```bash
git clone https://github.com/riccardovalencia/SUNSpinsInCavityQED.git
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
