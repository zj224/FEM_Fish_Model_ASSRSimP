# FEM Fish Model ASSR Simulation

This repository contains a Python implementation of a finite-element-inspired fish body model used for ASSR-style simulations. The code computes the eel/fish dynamics, integrates the state over time, and visualizes the motion.

## Overview

The project includes:

- A dynamic model of an articulated fish body
- Joint torque generation and passive damping terms
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

