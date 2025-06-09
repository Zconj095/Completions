from MPBR_Base import *
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from qiskit_aer import Aer
    from qiskit import QuantumCircuit, transpile, execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def analyze_habit_basics(frequencies):
    """Analyze generic habit data using BrainwaveAnalyzer."""
    analyzer = BrainwaveAnalyzer()

    # If cupy is available use GPU arrays
    if CUPY_AVAILABLE:
        frequencies = cp.asnumpy(cp.asarray(frequencies))

    results = analyzer.batch_analyze_frequencies(frequencies)
    return results


def demo_quantum_component():
    """Demo quantum computation if qiskit-aer is available."""
    if not QISKIT_AVAILABLE:
        print("Qiskit Aer not available; skipping quantum demo.")
        return

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    backend = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, backend)
    job = execute(compiled, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()
    print("Quantum demo counts:", counts)


if __name__ == "__main__":
    sample_freqs = [0.5, 4, 8, 13, 30]
    analysis = analyze_habit_basics(sample_freqs)
    print("Basic Habit Analysis:")
    print(analysis)
    demo_quantum_component()


try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def analyze_habit_by_development(start_data):
    """Analyze habits grouped by when they started developing."""
    analyzer = BrainwaveAnalyzer()
    results = {}

    for start_period, freqs in start_data.items():
        if CUPY_AVAILABLE:
            freqs = cp.asnumpy(cp.asarray(freqs))
        results[start_period] = analyzer.batch_analyze_frequencies(freqs)

    return results


if __name__ == "__main__":
    development_data = {
        "childhood": [4, 8, 8],
        "adulthood": [13, 13, 30],
    }
    analysis = analyze_habit_by_development(development_data)
    print("Habit Development Timeframe Analysis:")
    for start, res in analysis.items():
        print(f"Started: {start}")
        print(res)


try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def analyze_habit_by_origin(origin_map):
    """Analyze habits grouped by their origin location."""
    analyzer = BrainwaveAnalyzer()
    results = {}

    for origin, freqs in origin_map.items():
        if CUPY_AVAILABLE:
            freqs = cp.asnumpy(cp.asarray(freqs))
        results[origin] = analyzer.batch_analyze_frequencies(freqs)

    return results


if __name__ == "__main__":
    origin_data = {
        "home": [8, 13, 13],
        "work": [30, 30, 8],
    }
    analysis = analyze_habit_by_origin(origin_data)
    print("Habit Origin Analysis:")
    for loc, res in analysis.items():
        print(f"Location: {loc}")
        print(res)


try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def analyze_habit_by_repetition(frequencies):
    """Analyze habit data focusing on repetition counts."""
    analyzer = BrainwaveAnalyzer()

    if CUPY_AVAILABLE:
        frequencies = cp.asnumpy(cp.asarray(frequencies))

    repetition_counts = Counter(frequencies)
    repeated = [freq for freq, cnt in repetition_counts.items() for _ in range(cnt)]
    return analyzer.batch_analyze_frequencies(repeated)


if __name__ == "__main__":
    sample_freqs = [4, 4, 8, 8, 8, 13]
    result = analyze_habit_by_repetition(sample_freqs)
    print("Habit Repetition Analysis:")
    print(result)



try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from qiskit_aer import Aer
    from qiskit import QuantumCircuit, transpile, execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def analyze_habit_by_sequence(frequencies):
    """Analyze habit data sorted by frequency sequence."""
    analyzer = BrainwaveAnalyzer()

    if CUPY_AVAILABLE:
        frequencies = cp.asnumpy(cp.asarray(frequencies))

    frequencies = sorted(frequencies)
    return analyzer.batch_analyze_frequencies(frequencies)


if __name__ == "__main__":
    sample_freqs = [13, 8, 30, 0.5, 4]
    result = analyze_habit_by_sequence(sample_freqs)
    print("Habit Sequence Analysis:")
    print(result)



try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def analyze_habit_by_timeframe(data):
    """Analyze habits grouped by the timeframe they occurred."""
    analyzer = BrainwaveAnalyzer()
    results = {}

    for timeframe, freqs in data.items():
        if CUPY_AVAILABLE:
            freqs = cp.asnumpy(cp.asarray(freqs))
        results[timeframe] = analyzer.batch_analyze_frequencies(freqs)

    return results


if __name__ == "__main__":
    timeframe_data = {
        "morning": [8, 8, 13],
        "evening": [30, 4, 4],
    }
    analysis = analyze_habit_by_timeframe(timeframe_data)
    print("Habit Timeframe Analysis:")
    for tf, res in analysis.items():
        print(f"Timeframe: {tf}")
        print(res)
