should_print = True

from qto.problems.facility_location_problem import generate_flp
from qto.problems.set_cover_problem import generate_scp
from qto.problems.k_partition_problem import generate_kpp
from qto.problems.graph_coloring_problem import generate_gcp
from qto.model import LinearConstrainedBinaryOptimization as LcboModel
from qto.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from qto.solvers.qiskit import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver, RasenganSolver, RasenganSegmentedSolver,
    AerProvider, AerGpuProvider, DdsimProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, 
)
import numpy as np
import random
np.random.seed(0xdb)
random.seed(0x7f)

num_case = 1
# a, b = generate_scp(num_case,[(3, 3)])
a, b = generate_flp(num_case, [(4, 3)], 1, 20)
# a, b = generate_kpp(num_case, [(5, 3, 4)], 1, 20)
# a, b = generate_gcp(num_case, [(3, 2)])
# print(a[0][0])
# (1, [(2, 1), (3, 2), (3, 3), (4, 3), (4, 4)], 1, 20)

print(b)

best_lst = []
arg_lst = []

for i in range(num_case):
    opt = CobylaOptimizer(max_iter=5, save_address="save")
    aer = DdsimProvider()
    fake = FakeKyivProvider()
    gpu = AerGpuProvider()
    a[0][i].set_penalty_lambda(200)
    solver = RasenganSolver(
        prb_model=a[0][i],  # 问题模型
        optimizer=opt,  # 优化器
        provider=aer,  # 提供器（backend + 配对 pass_mannager ）
        num_layers=5,
        shots=1024,
        # num_segments=10,
        # mcx_mode="linear",
    )
    result = solver.solve()
    u, v, w, x = solver.evaluation()
    print(f"{i}: {u}, {v}, {w}, {x}")

    best_lst.append(u)
    arg_lst.append(w)
    print(aer.run_count)
    # print(solver.circuit_analyze(['depth', 'culled_depth', 'num_params']))
    # print(list(solver.time_analyze()))
    # print(sum(best_lst) / num_case, sum(arg_lst) / num_case)
    t1, t2 = solver.time_analyze()
    # print(counter.total_run_time )
    print("classical", t1)
    print("quantum", t2)
    # print(opt.cost_history)
