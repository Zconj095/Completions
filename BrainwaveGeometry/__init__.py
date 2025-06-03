import numpy as np
import sympy as sp
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy import special
import holoviews as hv
import tensorflow as tf
import os

# Constants
NDIM = 3
NPTS = 5000
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability

def relative_error(a, b):
    return np.linalg.norm(np.array(a) - np.array(b)) / (np.linalg.norm(np.array(a)) + 1e-12)

def distance_field(mesh, center=(0, 0, 0)):
    center = np.array(center)
    if len(center) < mesh.shape[1]:
        # Pad center with zeros if it has fewer dimensions than mesh
        center = np.pad(center, (0, mesh.shape[1] - len(center)))
    elif len(center) > mesh.shape[1]:
        # Truncate center if it has more dimensions than mesh
        center = center[:mesh.shape[1]]
    return np.linalg.norm(mesh - center, axis=1)

# Euclidean Manifold Class
class EuclideanManifold:
    def __init__(self, ndim: int = NDIM, npts: int = NPTS):
        self.ndim = ndim
        self.npts = npts
        self.points = self.generate_points()
        self.simplices = Delaunay(self.points).simplices
        self.metric = self.gen_metric()
        self.mesh = self.points  # For compatibility
        self.coords = sp.symbols('x y z')[:self.ndim]

    def gen_metric(self):
        # Identity metric for Euclidean space
        return np.eye(self.ndim)

    def generate_points(self):
        # Uniform random points in unit cube
        return np.random.rand(self.npts, self.ndim)

# Lorentz Transform Class
class LorentzTransform:
    def __init__(self, vel: float):
        self.vel = vel
        self.c = 299792458  # Speed of light in m/s
        self.beta = vel / self.c
        self.matrix = self.get_matrix()

    def get_matrix(self):
        # 1D Lorentz boost along x
        gamma = 1 / np.sqrt(1 - self.beta ** 2)
        return np.array([
            [gamma, -gamma * self.beta, 0, 0],
            [-gamma * self.beta, gamma, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def apply(self, u):
        return self.matrix @ u

    def inverse(self, u):
        return np.linalg.inv(self.matrix) @ u

class EuclideanExporter:
    def __init__(self, fields: dict):
        self.fields = fields
        self.timesteps = []
        self.export_dir = "sim_data"
        # Create directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)

    def export_fields(self, step: int):  # Remove @tf.function decorator
        E = self.fields['E']
        B = self.fields['B']
        
        # Convert TensorFlow tensors to NumPy if needed
        if hasattr(E, 'numpy'):
            E = E.numpy()
        if hasattr(B, 'numpy'):
            B = B.numpy()
        
        # Use NumPy save (recommended)
        np.save(f"{self.export_dir}/E_{step}.npy", E)
        np.save(f"{self.export_dir}/B_{step}.npy", B)
        
        self.timesteps.append(step)

    def render_timeline(self):
        # Create a comprehensive visualization
        if not self.timesteps:
            print("No timesteps to visualize")
            return None
            
        # Collect all field data
        all_E_data = []
        all_B_data = []
        
        for t in self.timesteps:
            try:
                E_data = np.load(f"{self.export_dir}/E_{t}.npy")
                B_data = np.load(f"{self.export_dir}/B_{t}.npy")
                
                # Calculate field magnitudes
                E_magnitude = np.linalg.norm(E_data, axis=1)
                B_magnitude = np.linalg.norm(B_data, axis=1)
                
                all_E_data.append(np.mean(E_magnitude))
                all_B_data.append(np.mean(B_magnitude))
                
            except FileNotFoundError:
                all_E_data.append(0)
                all_B_data.append(0)
        
        # Create time series plots
        E_curve = hv.Curve((self.timesteps, all_E_data), 
                          kdims=['Time Step'], vdims=['Average E-field'])
        B_curve = hv.Curve((self.timesteps, all_B_data), 
                          kdims=['Time Step'], vdims=['Average B-field'])
        
        # Create field magnitude plots for latest timestep
        if self.timesteps:
            latest_t = self.timesteps[-1]
            try:
                E_data = np.load(f"{self.export_dir}/E_{latest_t}.npy")
                B_data = np.load(f"{self.export_dir}/B_{latest_t}.npy")
                
                E_magnitude = np.linalg.norm(E_data, axis=1)
                B_magnitude = np.linalg.norm(B_data, axis=1)
                
                E_scatter = hv.Scatter((range(len(E_magnitude)), E_magnitude),
                                     kdims=['Point Index'], vdims=['E-field Magnitude'])
                B_scatter = hv.Scatter((range(len(B_magnitude)), B_magnitude),
                                     kdims=['Point Index'], vdims=['B-field Magnitude'])
                
                # Combine all plots
                layout = (E_curve + B_curve + E_scatter + B_scatter).cols(2)
                return layout
                
            except FileNotFoundError:
                layout = (E_curve + B_curve).cols(2)
                return layout
        
        return None

# Placeholder DEC operators (requires actual implementation)
class EuclideanDEC:
    def __init__(self, metric):
        self.metric = metric

    @property
    def d(self):
        # Placeholder for exterior derivative
        def grad(f):
            return np.gradient(f)
        return grad

    @property
    def codifferential(self):
        # Placeholder for codifferential
        def codiff(f):
            return -np.gradient(f)
        return codiff

    @property
    def hodge(self):
        # Placeholder for Hodge star
        def hodge_star(f):
            return f
        return hodge_star

    def maxwells_equations(self, E, B):
        # Placeholder Maxwell's equations
        dF = self.d(B) - self.d(E)
        return self.hodge(dF)

    def dirac_equation(self, psi, A):
        # Placeholder Dirac equation
        return psi + A

class AnalyticSolution:
    def __init__(self, euclidean: EuclideanManifold):
        self.coords = euclidean.coords
        self.t = sp.Symbol('t')

    def constant_e_field(self):
        Ex, Ey, Ez = 1, 0, 0
        B = sp.Matrix([0, 0, 0])
        return {'E': sp.Matrix([Ex, Ey, Ez]), 'B': B}

    def plane_wave(self, u, k):
        x, y, z = self.coords
        psi = u * sp.exp(sp.I * k * (x + y + z))
        return {'psi': psi}

    def verify(self, numeric_soln):
        analytic_soln = self.plane_wave(1 + 2j, 1 + 1j)
        psi_err = relative_error(analytic_soln['psi'], numeric_soln['psi'])
        print(f"Relative Error in psi: {psi_err}")

class MagneticPotential:
    def __init__(self, euclidean: EuclideanManifold):
        self.mesh = euclidean.mesh
        self.time = 0

    def solenoid(self, turns=10, radius=1, current=1):
        # Use 3D center to match mesh dimensions
        dist = distance_field(self.mesh, center=(0, 0, 0))
        Bz = (mu0 * turns * current) / (2 * radius ** 2)
        return Bz * (1 - (dist / radius) ** 2)

    def oscillating(self, freq=1, amplitude=1):
        return amplitude * np.sin(freq * self.time) * np.ones(self.mesh.shape[0])

    def update(self, time):
        self.time = time

class Simulation:
    def __init__(self):
        self.manifold = EuclideanManifold()
        self.potential = MagneticPotential(self.manifold)

    def run(self, steps: int, dt: float = 0.01):
        """Enhanced simulation runner with field calculations and visualization"""
        # Initialize field storage
        E_fields = []
        B_fields = []
        A_fields = []
        
        # Setup DEC operators
        dec = EuclideanDEC(self.manifold.metric)
        
        # Initialize exporter
        fields_dict = {'E': np.zeros((self.manifold.npts, 3)), 
                      'B': np.zeros((self.manifold.npts, 3))}
        exporter = EuclideanExporter(fields_dict)
        
        print(f"Starting simulation with {steps} steps, dt={dt}")
        
        for i in range(steps):
            current_time = i * dt
            
            # Update potential with time
            self.potential.update(current_time)
            
            # Calculate magnetic vector potential
            A_solenoid = self.potential.solenoid(turns=20, radius=0.5, current=10)
            A_oscillating = self.potential.oscillating(freq=2*np.pi, amplitude=0.1)
            A_total = A_solenoid + A_oscillating
            
            # Calculate magnetic field B = curl(A)
            B_field = dec.d(A_total)  # Using exterior derivative as curl
            if isinstance(B_field, (int, float)):
                B_field = np.full((self.manifold.npts, 3), B_field)
            elif B_field.ndim == 1:
                B_field = np.column_stack([B_field, np.zeros((len(B_field), 2))])
            
            # Calculate electric field E = -dA/dt (simplified)
            if i > 0:
                dA_dt = (A_total - A_fields[-1]) / dt
                E_field = -np.column_stack([dA_dt, np.zeros((len(dA_dt), 2))])
            else:
                E_field = np.zeros((self.manifold.npts, 3))
            
            # Store fields
            A_fields.append(A_total)
            B_fields.append(B_field)
            E_fields.append(E_field)
            
            # Update exporter fields
            fields_dict['E'] = E_field
            fields_dict['B'] = B_field
            
            # Export data more frequently for better visualization
            if i % 2 == 0:  # Export every 2 steps instead of 10
                exporter.export_fields(i)
                print(f"Step {i}: |B|_max = {np.max(np.linalg.norm(B_field, axis=1)):.4f}")
                print(f"Step {i}: |E|_max = {np.max(np.linalg.norm(E_field, axis=1)):.4f}")
        
        # Final analysis
        self._analyze_results(A_fields, B_fields, E_fields)
        
        # Generate and display visualization
        print("\nGenerating visualization...")
        timeline = exporter.render_timeline()
        if timeline:
            # Save or display the visualization
            try:
                hv.save(timeline, 'electromagnetic_fields.html')
                print("Visualization saved as 'electromagnetic_fields.html'")
            except Exception as e:
                print(f"Could not save visualization: {e}")
        
        return {'A': A_fields, 'B': B_fields, 'E': E_fields}
    
    def _analyze_results(self, A_fields, B_fields, E_fields):
        """Analyze simulation results"""
        print("\nSimulation Analysis:")
        print(f"Total energy at final step: {self._calculate_energy(E_fields[-1], B_fields[-1]):.6f}")
        print(f"Field variation over time: {np.std([np.mean(B) for B in B_fields]):.6f}")
        
        # Verify conservation laws
        energies = [self._calculate_energy(E, B) for E, B in zip(E_fields, B_fields)]
        energy_conservation = np.std(energies) / np.mean(energies) if np.mean(energies) > 0 else 0
        print(f"Energy conservation (std/mean): {energy_conservation:.6f}")

    def _calculate_energy(self, E_field, B_field):
        """Calculate total energy from E and B fields"""
        # Simplified energy calculation (placeholder)
        u_E = 0.5 * np.linalg.norm(E_field) ** 2
        u_B = 0.5 * np.linalg.norm(B_field) ** 2
        return u_E + u_B

# Usage example (commented out for module use)
if __name__ == "__main__":
    sim = Simulation()
    # Run longer simulation to see field evolution
    results = sim.run(50, dt=0.001)  # 50 steps with smaller time step
    
    # Optional: Access results for further analysis
    print(f"\nSimulation completed with {len(results['B'])} timesteps")
    print(f"Final B-field range: {np.min(results['B'][-1]):.6f} to {np.max(results['B'][-1]):.6f}")
