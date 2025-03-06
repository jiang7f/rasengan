from .provider import Provider, CORE_BASIS_GATES, EXTENDED_BASIS_GATES
from qiskit import QuantumCircuit
from typing import Dict
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from mqt.ddsim import DeterministicNoiseSimulator
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,phase_amplitude_damping_error,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from typing import List, Union
from mqt.ddsim.deterministicnoisesimulator import DeterministicNoiseSimulatorBackend


class NoiseDDsimProvider(Provider):
    def __init__(self,**kwargs):
        super().__init__()
        self.p_gate1 = kwargs.get('p_gate1',0.001)
        self.sim_noise = DeterministicNoiseSimulatorBackend()
        
    def get_counts(self, qc: QuantumCircuit, shots: int) -> Dict:
        # Create noisy simulator backend
        result = self.sim_noise.run(
            qc,
            shots=shots,
            noise_effects="AP",
            noise_probability=self.p_gate1,
            amp_damping_probability=0.002,
            multi_qubit_gate_factor=2,
            # simulator_seed=i,
        ).result()
        counts = result.get_counts(0)

        return counts
    
    def get_probabilities(self, qc: QuantumCircuit, shots: int) -> Dict:
        
        counts = self.get_counts(qc, shots)
        probabilities = {}
        for key, value in counts.items():
            probabilities[key] = value / shots
        return probabilities


    def transpile(self, qc: QuantumCircuit) -> QuantumCircuit:
        
        circ_tnoise = transpile(qc, self.sim_noise)
        return circ_tnoise
