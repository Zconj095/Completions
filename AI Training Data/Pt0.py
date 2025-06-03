"""
Enhanced NumPy Tutorial: Comprehensive Guide to Arrays and Matrices
Author: GitHub Copilot
Version: 2.0 - Enhanced Edition
"""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time
import warnings
from typing import Union, Tuple, List

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 0. SETUP AND CONFIGURATION
# =============================================================================

print("=== NUMPY CONFIGURATION ===")
print(f"NumPy version: {np.__version__}")
print(f"NumPy configuration:")
np.show_config()
print(f"Available CPU features: {np.show_runtime()}")

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. VECTOR AND MATRIX CREATION
# =============================================================================

print("\n=== VECTOR AND MATRIX CREATION ===")

# Create vectors
vector_row = np.array([1, 2, 3])
vector_column = np.array([[1], [2], [3]])

print(f"Row vector: {vector_row}")
print(f"Column vector shape: {vector_column.shape}")

# Create matrices using different methods
matrix_basic = np.array([[1, 2], [3, 4], [5, 6]])
matrix_zeros = np.zeros((3, 3))
matrix_ones = np.ones((2, 4))
matrix_identity = np.eye(3)
matrix_full = np.full((2, 3), 7)

# Advanced creation methods
matrix_linspace = np.linspace(0, 10, 12).reshape(3, 4)
matrix_logspace = np.logspace(1, 3, 6).reshape(2, 3)
matrix_random = np.random.rand(3, 3)
matrix_arange = np.arange(1, 13).reshape(3, 4)

print(f"Basic matrix:\n{matrix_basic}")
print(f"Identity matrix:\n{matrix_identity}")
print(f"Linspace matrix:\n{matrix_linspace}")
print(f"Random matrix:\n{matrix_random}")

# Data type specifications
matrix_int32 = np.zeros((2, 2), dtype=np.int32)
matrix_float64 = np.ones((2, 2), dtype=np.float64)
matrix_complex = np.zeros((2, 2), dtype=np.complex128)

print(f"Int32 matrix dtype: {matrix_int32.dtype}")
print(f"Complex matrix:\n{matrix_complex}")

# =============================================================================
# 2. SPARSE MATRICES
# =============================================================================

print("\n=== SPARSE MATRICES ===")

# Create different types of sparse matrices
dense_matrix = np.array([[0, 0, 3], [0, 4, 0], [5, 0, 0]])
csr_matrix = sparse.csr_matrix(dense_matrix)  # Compressed Sparse Row
csc_matrix = sparse.csc_matrix(dense_matrix)  # Compressed Sparse Column
coo_matrix = sparse.coo_matrix(dense_matrix)  # Coordinate format

print(f"Dense matrix:\n{dense_matrix}")
print(f"CSR sparse matrix:\n{csr_matrix}")
print(f"Memory efficiency: {csr_matrix.nnz}/{csr_matrix.size} non-zero elements")
print(f"Sparsity: {(1 - csr_matrix.nnz/csr_matrix.size)*100:.1f}%")

# Sparse matrix operations
sparse_sum = csr_matrix + csc_matrix
print(f"Sparse matrix addition result:\n{sparse_sum.toarray()}")

# =============================================================================
# 3. ARRAY INDEXING AND SLICING
# =============================================================================

print("\n=== INDEXING AND SLICING ===")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Advanced indexing examples
print(f"Original array: {arr}")
print(f"Every 2nd element: {arr[::2]}")
print(f"Reverse array: {arr[::-1]}")
print(f"Boolean indexing (>5): {arr[arr > 5]}")
print(f"Fancy indexing: {arr[[0, 2, 4, 6]]}")

# Matrix indexing
print(f"\nMatrix:\n{matrix}")
print(f"Anti-diagonal: {matrix[range(3), range(2, -1, -1)]}")
print(f"Upper triangle: {matrix[np.triu_indices(3)]}")
print(f"Conditional selection: {matrix[matrix % 2 == 0]}")

# Multi-dimensional indexing
arr_3d = np.random.randint(0, 10, (2, 3, 4))
print(f"3D array shape: {arr_3d.shape}")
print(f"First 'page': {arr_3d[0]}")

# =============================================================================
# 4. ARRAY PROPERTIES AND INFORMATION
# =============================================================================

print("\n=== ARRAY PROPERTIES ===")

sample_array = np.random.rand(3, 4, 2)
print(f"Shape: {sample_array.shape}")
print(f"Size: {sample_array.size}")
print(f"Dimensions: {sample_array.ndim}")
print(f"Data type: {sample_array.dtype}")
print(f"Memory usage: {sample_array.nbytes} bytes")
print(f"Item size: {sample_array.itemsize} bytes")
print(f"Memory layout - C contiguous: {sample_array.flags['C_CONTIGUOUS']}")
print(f"Memory layout - Fortran contiguous: {sample_array.flags['F_CONTIGUOUS']}")

# Array info function
def array_info(arr: np.ndarray) -> None:
        """Print comprehensive array information."""
        print(f"Array info:")
        print(f"  Shape: {arr.shape}")
        print(f"  Data type: {arr.dtype}")
        print(f"  Memory: {arr.nbytes} bytes")
        print(f"  Min/Max: {arr.min():.3f}/{arr.max():.3f}")

# =============================================================================
# 5. MATHEMATICAL OPERATIONS
# =============================================================================

print("\n=== MATHEMATICAL OPERATIONS ===")

# Element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")
print(f"Power: {a ** 2}")
print(f"Modulo: {b % a}")

# Mathematical functions
print(f"Square root: {np.sqrt(a)}")
print(f"Exponential: {np.exp(a)}")
print(f"Logarithm: {np.log(a)}")
print(f"Trigonometric (sin): {np.sin(a)}")

# Broadcasting examples
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
scalar = 5

print(f"Matrix + vector (broadcasting):\n{matrix + vector}")
print(f"Matrix * scalar (broadcasting):\n{matrix * scalar}")

# Complex operations
complex_arr = np.array([1+2j, 3+4j, 5+6j])
print(f"Complex array: {complex_arr}")
print(f"Real parts: {np.real(complex_arr)}")
print(f"Imaginary parts: {np.imag(complex_arr)}")
print(f"Magnitude: {np.abs(complex_arr)}")

# =============================================================================
# 6. STATISTICAL FUNCTIONS
# =============================================================================

print("\n=== STATISTICAL FUNCTIONS ===")

# Generate sample data
data = np.random.normal(50, 15, (1000,))
matrix_data = np.random.rand(5, 4)

print(f"Descriptive statistics:")
print(f"  Mean: {np.mean(data):.2f}")
print(f"  Median: {np.median(data):.2f}")
print(f"  Standard deviation: {np.std(data):.2f}")
print(f"  Variance: {np.var(data):.2f}")
print(f"  Min/Max: {np.min(data):.2f}/{np.max(data):.2f}")
print(f"  Range: {np.ptp(data):.2f}")
print(f"  Percentiles (25%, 50%, 75%): {np.percentile(data, [25, 50, 75])}")

# Axis-wise operations
print(f"\nMatrix statistics (axis-wise):")
print(f"  Column means: {np.mean(matrix_data, axis=0)}")
print(f"  Row means: {np.mean(matrix_data, axis=1)}")
print(f"  Column std: {np.std(matrix_data, axis=0)}")

# Correlation and covariance
sample1 = np.random.normal(0, 1, 100)
sample2 = sample1 + np.random.normal(0, 0.5, 100)
correlation = np.corrcoef(sample1, sample2)[0, 1]
print(f"Correlation coefficient: {correlation:.3f}")

# =============================================================================
# 7. ARRAY MANIPULATION
# =============================================================================

print("\n=== ARRAY MANIPULATION ===")

# Reshaping and transformations
original = np.arange(24)
print(f"Original: {original}")
print(f"Reshaped (3x8):\n{original.reshape(3, 8)}")
print(f"Reshaped (4x6):\n{original.reshape(4, 6)}")
print(f"Transposed:\n{original.reshape(4, 6).T}")

# Advanced reshaping
print(f"Flattened: {original.reshape(4, 6).flatten()}")
print(f"Raveled: {original.reshape(4, 6).ravel()}")

# Stacking and splitting
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

print(f"Horizontal stack: {np.hstack([arr1, arr2, arr3])}")
print(f"Vertical stack:\n{np.vstack([arr1, arr2, arr3])}")
print(f"Depth stack shape: {np.dstack([arr1, arr2, arr3]).shape}")

# Splitting arrays
large_array = np.arange(12)
split_arrays = np.array_split(large_array, 3)
print(f"Split into 3 parts: {split_arrays}")

# Concatenation
concat_result = np.concatenate([arr1, arr2, arr3])
print(f"Concatenated: {concat_result}")

# =============================================================================
# 8. LINEAR ALGEBRA OPERATIONS
# =============================================================================

print("\n=== LINEAR ALGEBRA ===")

# Matrix operations
A = np.array([[2, 1], [1, 2]])
B = np.array([[1, 0], [0, 1]])
C = np.array([[3, 1], [2, 4]])

print(f"Matrix A:\n{A}")
print(f"Matrix multiplication A @ B:\n{A @ B}")
print(f"Element-wise multiplication A * B:\n{A * B}")
print(f"Matrix power A^2:\n{np.linalg.matrix_power(A, 2)}")

# Advanced linear algebra
print(f"Determinant of A: {np.linalg.det(A):.2f}")
print(f"Trace of A: {np.trace(A)}")
print(f"Rank of A: {np.linalg.matrix_rank(A)}")

# Matrix decompositions
eigenvals, eigenvecs = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

# SVD decomposition
U, s, Vt = np.linalg.svd(A)
print(f"SVD singular values: {s}")

# Solving linear systems
x = np.linalg.solve(A, np.array([3, 4]))
print(f"Solution to Ax = [3, 4]: {x}")

# Matrix inverse and pseudo-inverse
try:
        A_inv = np.linalg.inv(A)
        print(f"Inverse of A:\n{A_inv}")
except np.linalg.LinAlgError:
        print("Matrix is singular, cannot compute inverse")

# =============================================================================
# 9. ADVANCED OPERATIONS
# =============================================================================

print("\n=== ADVANCED OPERATIONS ===")

# Conditional operations
data = np.random.randint(1, 10, (4, 4))
print(f"Original data:\n{data}")
print(f"Where > 5:\n{np.where(data > 5, data, 0)}")
print(f"Clip values (2-7):\n{np.clip(data, 2, 7)}")

# Advanced selection
mask = (data > 3) & (data < 8)
print(f"Boolean mask:\n{mask}")
print(f"Masked values: {data[mask]}")

# Sorting and searching
print(f"Sorted (axis=0):\n{np.sort(data, axis=0)}")
print(f"Argsort indices: {np.argsort(data.flatten())}")
print(f"Unique values: {np.unique(data)}")

# Set operations
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])
print(f"Intersection: {np.intersect1d(arr1, arr2)}")
print(f"Union: {np.union1d(arr1, arr2)}")
print(f"Difference: {np.setdiff1d(arr1, arr2)}")

# Histogram and binning
values = np.random.normal(0, 1, 1000)
hist, bin_edges = np.histogram(values, bins=20)
print(f"Histogram bins: {len(bin_edges)-1}")
print(f"Max frequency: {hist.max()}")

# =============================================================================
# 10. RANDOM NUMBER GENERATION
# =============================================================================

print("\n=== RANDOM NUMBER GENERATION ===")

# Set up random generator
rng = np.random.default_rng(42)

# Different distributions
normal_sample = rng.normal(0, 1, 5)
uniform_sample = rng.uniform(0, 1, 5)
binomial_sample = rng.binomial(10, 0.5, 5)
poisson_sample = rng.poisson(3, 5)
exponential_sample = rng.exponential(2, 5)

print(f"Normal distribution: {normal_sample}")
print(f"Uniform distribution: {uniform_sample}")
print(f"Binomial distribution: {binomial_sample}")
print(f"Poisson distribution: {poisson_sample}")
print(f"Exponential distribution: {exponential_sample}")

# Random sampling and permutations
population = np.arange(100)
sample = rng.choice(population, size=10, replace=False)
shuffled = rng.permutation(np.arange(10))

print(f"Random sample: {sample}")
print(f"Random permutation: {shuffled}")

# Random matrix generation
random_matrix = rng.random((3, 3))
random_integers = rng.integers(1, 10, (3, 3))

print(f"Random matrix:\n{random_matrix}")
print(f"Random integers:\n{random_integers}")

# =============================================================================
# 11. PERFORMANCE OPTIMIZATION
# =============================================================================

print("\n=== PERFORMANCE OPTIMIZATION ===")

def time_operation(func, *args, iterations=3):
        """Time a function execution."""
        times = []
        for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args)
                end = time.perf_counter()
                times.append(end - start)
        return np.mean(times), result

# Vectorized vs loop comparison
n = 100000
data = np.random.rand(n)

# Vectorized operation
def vectorized_sum_squares(arr):
        return np.sum(arr ** 2)

# Python loop (for comparison - don't use this!)
def loop_sum_squares(arr):
        total = 0
        for x in arr:
                total += x ** 2
        return total

vec_time, vec_result = time_operation(vectorized_sum_squares, data)
print(f"Vectorized operation: {vec_time:.6f} seconds")

# Memory efficient operations
def memory_efficient_operation():
        # Use views instead of copies when possible
        large_array = np.random.rand(1000, 1000)
        subset = large_array[::10, ::10]  # This is a view
        return subset.mean()

mem_time, mem_result = time_operation(memory_efficient_operation)
print(f"Memory efficient operation: {mem_time:.6f} seconds")

# Broadcasting efficiency
matrix = np.random.rand(1000, 1000)
vector = np.random.rand(1000)

def efficient_broadcast():
        return matrix + vector  # Broadcasting

def inefficient_broadcast():
        return matrix + vector.reshape(-1, 1).repeat(1000, axis=1)  # Manual tiling

eff_time, _ = time_operation(efficient_broadcast)
print(f"Efficient broadcasting: {eff_time:.6f} seconds")

# =============================================================================
# 12. MEMORY MANAGEMENT AND BEST PRACTICES
# =============================================================================

print("\n=== MEMORY MANAGEMENT ===")

# View vs Copy demonstration
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]  # This is a view
copy = original[1:4].copy()  # This is a copy

print("Original array shares memory with view:")
print(f"  Original: {original}")
print(f"  View: {view}")
print(f"  Shares memory: {np.shares_memory(original, view)}")
print(f"  Copy shares memory: {np.shares_memory(original, copy)}")

# Modify view to show shared memory
view[0] = 999
print(f"After modifying view: {original}")

# Memory layout optimization
arr_c = np.random.rand(1000, 1000)  # C-contiguous by default
arr_f = np.asfortranarray(arr_c)    # Convert to Fortran-contiguous

print(f"C-contiguous access time:")
c_time, _ = time_operation(lambda: arr_c.sum(axis=0))
print(f"  Column sum: {c_time:.6f} seconds")

print(f"Fortran-contiguous access time:")
f_time, _ = time_operation(lambda: arr_f.sum(axis=0))
print(f"  Column sum: {f_time:.6f} seconds")

# =============================================================================
# 13. DATA TYPES AND PRECISION
# =============================================================================

print("\n=== DATA TYPES AND PRECISION ===")

# Different numeric types
int8_arr = np.array([1, 2, 3], dtype=np.int8)
float32_arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
float64_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)

print(f"Int8 array: {int8_arr} (memory: {int8_arr.nbytes} bytes)")
print(f"Float32 array: {float32_arr} (memory: {float32_arr.nbytes} bytes)")
print(f"Float64 array: {float64_arr} (memory: {float64_arr.nbytes} bytes)")

# Type conversion
converted = int8_arr.astype(np.float64)
print(f"Converted to float64: {converted}")

# Precision comparison
small_value = np.array([1e-8], dtype=np.float32)
large_value = np.array([1.0], dtype=np.float32)
result = large_value + small_value
print(f"Precision test (float32): 1.0 + 1e-8 = {result[0]}")

# =============================================================================
# 14. ERROR HANDLING AND VALIDATION
# =============================================================================

print("\n=== ERROR HANDLING ===")

def safe_divide(a, b):
        """Safely divide arrays with error handling."""
        try:
                with np.errstate(divide='ignore', invalid='ignore'):
                        result = np.divide(a, b)
                        # Replace inf and nan with appropriate values
                        result[np.isinf(result)] = 0
                        result[np.isnan(result)] = 0
                        return result
        except Exception as e:
                print(f"Error in division: {e}")
                return None

# Test safe division
a = np.array([1, 2, 3, 4])
b = np.array([2, 0, 1, 2])
safe_result = safe_divide(a, b)
print(f"Safe division result: {safe_result}")

# =============================================================================
# 15. PRACTICAL EXAMPLES
# =============================================================================

print("\n=== PRACTICAL EXAMPLES ===")

# Image processing simulation
def create_simple_image():
        """Create a simple synthetic image."""
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2) / 10)
        return Z

image = create_simple_image()
print(f"Image shape: {image.shape}")
print(f"Image statistics: min={image.min():.3f}, max={image.max():.3f}")

# Signal processing example
def generate_signal():
        """Generate a noisy sine wave."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        noise = np.random.normal(0, 0.1, 1000)
        noisy_signal = signal + noise
        return t, signal, noisy_signal

t, clean_signal, noisy_signal = generate_signal()
print(f"Signal SNR: {10 * np.log10(np.var(clean_signal) / np.var(noisy_signal - clean_signal)):.2f} dB")

# Financial data simulation
def simulate_stock_prices(days=252, initial_price=100):
        """Simulate stock price using random walk."""
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = initial_price * np.exp(np.cumsum(returns))
        return prices

stock_prices = simulate_stock_prices()
print(f"Stock simulation: Start=${stock_prices[0]:.2f}, End=${stock_prices[-1]:.2f}")
print(f"Max drawdown: {((stock_prices.min() / stock_prices[:np.argmin(stock_prices)].max()) - 1) * 100:.1f}%")

print("\n=== ENHANCED TUTORIAL COMPLETE ===")
print("This tutorial covered:")
print("• Advanced array creation and manipulation")
print("• Performance optimization techniques")
print("• Memory management best practices")
print("• Error handling and validation")
print("• Practical examples and applications")
print("• Comprehensive statistical operations")
print("• Linear algebra and matrix operations")
