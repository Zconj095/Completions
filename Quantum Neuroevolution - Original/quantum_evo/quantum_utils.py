"""Quantum circuit utilities using Qiskit Aer."""

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator


def classical_to_quantum_data(data_vector):
    """Encode a classical vector into a quantum state."""
    num_qubits = len(data_vector).bit_length() - 1
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.h(i)
    return circuit


def quantum_to_classical_data(circuit):
    """Measure a quantum state and decode the result."""
    simulator = AerSimulator()
    circuit.measure_all()
    transpiled = transpile(circuit, simulator)
    qobj = assemble(transpiled, shots=1)
    result = simulator.run(qobj).result()
    counts = result.get_counts(circuit)
    output_vector = max(counts, key=counts.get)
    return int(output_vector, 2)
