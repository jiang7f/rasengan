from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

# 参数设置
theta = 0.001  # 小角度，对应于 a = sin(theta)
num_iterations = int(10000/theta)  # 放大迭代次数

# 创建量子电路
qc = QuantumCircuit(1)

# 初始化：用 Ry 旋转构造 a|1> + b|0>
qc.ry(2 * theta, 0)

# 定义 oracle: 对 |1> 加一个 Z，相当于相位反转
def oracle(qc):
    qc.z(0)

# 定义扩散操作: 关于初始态反射
def diffuser(qc, theta):
    qc.ry(-2 * theta, 0)
    qc.z(0)
    qc.ry(2 * theta, 0)

# 应用多次振幅放大
for _ in range(num_iterations):
    oracle(qc)
    diffuser(qc, theta)

# 测量
qc.measure_all()

# 模拟
sampler = Sampler(mode=AerSimulator())
job = sampler.run([qc], shots=1024)

# 获取结果
result = job.result()
pub_result = result[0]

# 提取每次测量的结果
counts = pub_result.data.meas.get_counts()
print("Measurement counts:", counts)