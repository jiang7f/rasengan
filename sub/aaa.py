from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

import matplotlib.pyplot as plt

# 参数设置
theta = 0  # |1⟩ 的初始振幅 ≈ 0.001
shots = 1024
threshold = 0.5  # 目标成功概率阈值

# 定义 oracle 和 diffuser
def oracle(qc):
    qc.z(0)

def diffuser(qc, theta):
    qc.ry(-2 * theta, 0)
    qc.z(0)
    qc.ry(2 * theta, 0)

# 执行一次放大并测量 |1⟩ 的概率
def run_amplification(k, theta):
    qc = QuantumCircuit(1)
    qc.ry(2 * theta, 0)
    for _ in range(k):
        oracle(qc)
        diffuser(qc, theta)
    qc.measure_all()
    
    simulator = AerSimulator()
    sampler = Sampler(mode=simulator)
    job = sampler.run([qc], shots=shots)
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    
    count_1 = counts.get('1', 0)
    prob_1 = count_1 / shots
    return prob_1, counts

# 自适应迭代：k = 1, 2, 4, 8, ...
k = 1
adaptive_probs = []
adaptive_ks = []
found = False

while not found and k < 100000:
    prob, counts = run_amplification(k, theta)
    adaptive_probs.append(prob)
    adaptive_ks.append(k)
    print(f"Iter {k:>5}: P(|1⟩) = {prob:.4f} — Counts: {counts}")
    if prob >= threshold:
        found = True
    else:
        k *= 2

# 画图：每个 k 的概率
plt.figure(figsize=(8, 5))
plt.plot(adaptive_ks, adaptive_probs, marker='o', label='P(|1⟩)')
plt.axhline(threshold, color='gray', linestyle='--', label='Threshold = 0.5')
plt.xscale('log')
plt.xlabel('Number of Amplification Iterations (k)')
plt.ylabel('Probability of Measuring |1⟩')
plt.title('Adaptive Amplitude Amplification')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
