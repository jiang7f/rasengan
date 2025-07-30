import json
import random
import numpy as np
from tqdm import tqdm

from rasengan.problems.facility_location_problem import generate_flp
from rasengan.problems.set_cover_problem import generate_scp
from rasengan.problems.k_partition_problem import generate_kpp
from rasengan.problems.graph_coloring_problem import generate_gcp
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import DdsimProvider
from rasengan.solvers.qiskit.explorer import QtoExplorer

np.random.seed(0x7f)
random.seed(0x7f)

num_cases = 1

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)])

problems_pkg = [flp_problems_pkg, kpp_problems_pkg, scp_problems_pkg, gcp_problems_pkg]

space_list = []

for problem in tqdm(problems_pkg, desc="Analysing space"):
    problem_space_list = []
    for benchmark in tqdm(problem, desc="    processing", leave=False):
        opt = CobylaOptimizer(max_iter=200)
        aer = DdsimProvider()
        benchmark[0].set_penalty_lambda(200)
        solver = QtoExplorer(
            prb_model=benchmark[0],
            optimizer=opt,
            provider=aer,
            shots=1000,
        )
        exp_list, _, _ = solver.explore()
        problem_space_list.append(exp_list)
    space_list.append(problem_space_list)

with open("space_explore.json", "w") as f:
    json.dump(space_list, f)