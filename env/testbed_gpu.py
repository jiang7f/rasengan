should_print = True

from rasengan.model import LinearConstrainedBinaryOptimization as LcboModel
from rasengan.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from rasengan.solvers.qiskit import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver, RasenganSolver,
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

# model ----------------------------------------------
m = LcboModel()
x = m.addVars(3, name="x")
m.setObjective(x[0] + x[1] - x[2], "max")

m.addConstr(x[0] + x[1] + x[2] == 2)

print(m.lin_constr_mtx)
print(m)
optimize = m.optimize()
print(f"optimize_cost: {optimize}\n\n")
# sovler ----------------------------------------------
opt = CobylaOptimizer(max_iter=200)
gpu = AerGpuProvider()
aer = DdsimProvider()
solver = RasenganSolver(
    prb_model=m,  # 问题模型
    optimizer=opt,  # 优化器
    provider=gpu,  # 提供器（backend + 配对 pass_mannager ）
    num_layers=1,
    # mcx_mode="linear",
)
try:
    print(solver.circuit_analyze(['depth', 'width', 'culled_depth', 'num_one_qubit_gates']))
    result = solver.solve()
    eval = solver.evaluation()
    print(result)
    print(eval)
    print("="*50)
    print("✅  GPU environment configured successfully!")
    print("="*50)
except Exception as e:
    print("="*50)
    print("❌  GPU environment configuration failed.")
    print(e)
    print("="*50)

