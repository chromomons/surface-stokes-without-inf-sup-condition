import numpy as np
from ngsolve import SetNumThreads
import json

from utils import math_dict_to_cfs, printbf
from solvers import convergence_torus
from exact import Exact
from ngsolve import TaskManager


input_file_name = "./input_torus.json"

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

# text output of errors, separated by & for latex tables
txt_out = out_params['txt_out']
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

printbf(f"Using the P{order_u}-P{order_p} velocity-pressure pair on the torus.")
print()

l2us, h1us, l2ps, h1ps = [], [], [], []

msg = f"& l &   dof    & rate &   l2u    & rate &   h1u    & rate &   l2p    & rate &   h1p    "
printbf(msg)

if txt_out:
    fe = open(f"./output/txt/torus_p{order_u}-p{order_p}.txt", "w")
    fe.write(f"{msg}\n")
    fe.close()

hmax = 0.5

for nref in range(max_nref+1):
    h = hmax * 2**(-nref)
    phi = exact.cfs['phi']

    ndof, l2u, h1u, l2p, h1p = convergence_torus(
        maxh=h,
        nref=nref+1,
        exact=exact,
        order_u=order_u,
        order_p=order_p,
        logs=solver_logs
    )

    if len(l2us) > 0:
        msg = f"& {nref+1} & {ndof:.2E} & {np.log2(l2us[-1] / l2u):.2f} & {l2u:.2E} & {np.log2(h1us[-1] / h1u):.2f} & {h1u:.2E} & {np.log2(l2ps[-1] / l2p):.2f} & {l2p:.2E} & {np.log2(h1ps[-1] / h1p):.2f} & {h1p:.2E}"
    else:
        msg = f"& {nref+1} & {ndof:.2E} &      & {l2u:.2E} &      & {h1u:.2E} &      & {l2p:.2E} &      & {h1p:.2E}"

    l2us.append(l2u)
    h1us.append(h1u)
    l2ps.append(l2p)
    h1ps.append(h1p)

    # OUTPUT

    print(msg)

    if txt_out:
        fe = open(f"./output/txt/torus_p{order_u}-p{order_p}.txt", "a")
        fe.write(f"{msg}\n")
        fe.close()

