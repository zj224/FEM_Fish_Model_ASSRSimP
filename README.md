# ASSRSimP: FEM Model of an Eel-Inspired Soft Robot

This is the code for the paper "Modeling and Control of an Eel-Inspired Soft Robot for Design Optimization" by Zhangjingyi Jiang and Mark Campbell, published at IEEE CASE 2024 ([DOI: 10.1109/CASE59546.2024.10711519](https://doi.org/10.1109/CASE59546.2024.10711519)).

This repository contains a Python implementation of a finite-element model of an eel-inspired (anguilliform) soft robot with distributed torque actuators and hydrodynamic drag. The code computes the eel/fish dynamics, integrates the state over time, and visualizes the motion.

## Overview

The project includes:

- A finite-element model of an articulated, anguilliform soft robot body
- Distributed joint torque actuators with passive stiffness and damping terms
- Hydrodynamic drag and body inertia contributions
- Time integration of the system state
- Visualization of the fish motion

The main entry point is:

- `main_fish.py`

Supporting functions are in:

- `Fish_Functions/get_nancy_params.py`
- `Fish_Functions/step.py`
- `Fish_Functions/plot.py`
- `Fish_Functions/animate_eel.py`

## Requirements

This project uses Python and the following packages:

- numpy
- scipy
- numba
- matplotlib
- pytictoc

## Installation

It is recommended to use a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the simulation

From the project root:

```bash
python3 main_fish.py
```

This will run the simulation and generate the animation output for the fish model.

## Related work

This model is also the robot model used by [REACH (Real-time Estimator of Actuator Control and Health)](https://github.com/zj224/REACH_Fault_Estimation), published at IEEE RoboSoft 2025.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{jiang2024modeling,
  title     = {Modeling and Control of an Eel-Inspired Soft Robot for Design Optimization},
  author    = {Jiang, Zhangjingyi and Campbell, Mark},
  booktitle = {2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)},
  pages     = {2471--2476},
  year      = {2024},
  doi       = {10.1109/CASE59546.2024.10711519}
}
```

## Acknowledgments

This work was supported by ONR MURI grant N00014-22-1-2595.
