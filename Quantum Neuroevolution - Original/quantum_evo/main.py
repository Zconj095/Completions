"""Entry point for the Neuroquantum Evolution Toolkit."""

import cupy as cp
from .evolution import run_evolution
from .integration import integrate
from .visualize import plot_hfp


def main():
    data_vector = cp.ones(4)
    hfp_values = []
    for _ in range(3):
        hfp = integrate(data_vector)
        hfp_values.append(cp.asnumpy(hfp))
    winner = run_evolution()
    plot = plot_hfp(hfp_values)
    plot.show()
    print("Evolution complete. Winner genome:", winner)


if __name__ == "__main__":
    main()
