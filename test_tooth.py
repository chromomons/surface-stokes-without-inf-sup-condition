import numpy as np
from ngsolve import SetNumThreads
import json

from utils import math_dict_to_cfs, printbf
from solvers import figure_tooth
from exact import Exact
from ngsolve import TaskManager


input_file_name = "./input_tooth.json"

printbf("\nSolving the surface Stokes problem with non-inf-sup stable\n"
        "velocity-pressure pairs using elliptic reformulation.\n"
        "\n"
        "(c) Mansur Shakipov, 2025.\n")

f = open(input_file_name)

# loading JSON input files
args = json.load(f)

# Setting number of threads
SetNumThreads(args['num_threads'])

path_to_math_json = args['path_to_math_json']
f_math = open(path_to_math_json)
args_math = json.load(f_math)

# orders: space order only in this case
orders = args['orders']
order_u = orders['order_u']
order_p = orders['order_p']

# Meshing parameters
meshing_params = args['meshing_params']
# Output flags
out_params = args['out_params']

# Unpacking some of the parameters

# vtk_out for VTK output
vtk_out = out_params['vtk_out']
# solver logs (i.e. how much time is spend in assembly and linear solver)
solver_logs = out_params['solver_logs']

# number of refinements
max_nref = meshing_params['max_nref']

# EXACT SOLUTION

# Collecting all parameters for coefficient functions
exact_params = {}

# Creating Exact object
exact = Exact(name="stokes_torus", params=exact_params)
cfs = math_dict_to_cfs(args_math, exact_params)
exact.set_cfs(cfs)

printbf(f"Using the P{order_u}-P{order_p} velocity-pressure pair on the tooth surface.")
print()

l2us, h1us, l2ps, h1ps = [], [], [], []

hmax = 0.5

for nref in range(max_nref+1):
    h = hmax * 2**(-nref)
    phi = exact.cfs['phi']

    print(f"Refinement level: {nref+1}.")

    figure_tooth(
        maxh=h,
        nref=nref+1,
        exact=exact,
        order_u=order_u,
        order_p=order_p,
        logs=solver_logs
    )

