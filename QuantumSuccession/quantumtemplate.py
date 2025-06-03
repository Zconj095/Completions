import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate timeseries data
time_points = np.arange(0, 100)
series = np.sin(time_points) + np.random.randn(len(time_points)) * 0.1

# Reshape into samples with time feature
X = time_points.reshape(-1, 1) 
y = series

# Train SVM regression 
svr = SVR(kernel='rbf', C=100, gamma=0.1) 
svr.fit(X, y)

# Predict on a expanded time range 
X_predict = np.arange(0, 150).reshape(-1, 1)
y_predict = svr.predict(X_predict)

# Plot original and predicted time series
plt.plot(time_points, series, label='Original Time Series')  
plt.plot(X_predict.ravel(), y_predict, label='SVM Predicted Series')  
plt.legend()
plt.show()

import numpy as np

# Characteristic Energies 
solar_flux = 1000 # W/m^2. Natural light 
laser_flux = 5e5  # W/m^2. Artificial light

kT = 0.025        # 25 meV thermal energy at 300K 

# Subatomic Particles 

# Electron 
me_kg = 9.1e-31   # Electron rest mass  
me_joules = 0.5*me_kg*(3e8)**2  

# Proton
mp_kg = 1.67e-27  
mp_joules = mp_kg*(3e8)**2

# Ratios
laser_to_solar = laser_flux/solar_flux   # Artificial to natural light

light_to_thermal = solar_flux/(kT) # Photon energy to thermal (meV per area)

proton_to_photon = mp_joules/(laser_flux) # Proton rest energy to artificial photons

# Print comparisons 
print(f'Laser to Solar Ratio: {laser_to_solar/1e6:.2f} Million')  

print(f'Photon to Thermal Ratio: {light_to_thermal:.2e}')

print(f'Proton to Photon Ratio: {proton_to_photon:.2f}')


import numpy as np

# Sound and EM properties
sound_speed = 343    # m/s 
em_freq = 5e9        # Hz

# Define wavefunctions  
def sound_wave(x, t):
    k = 2 * np.pi / sound_speed
    w = 2 * np.pi * em_freq
    phi = 0 # Phase offset
     
    return np.cos(k * x - w * t + phi) 

def light_wave(x, t):
    w = 2 * np.pi * em_freq  # Define w for light_wave, assuming same frequency as sound_wave for simplicity
    k = w / em_freq
    phi = np.pi / 4  
    
    return np.cos(k * x - w * t + phi)

# Position and time
x = 0
t = 0
 
# Calculate initial waves 
sound = sound_wave(x, t)
light = light_wave(x, t)  

# Subinitiate: Prioritize sound component to 1  
if abs(sound) >= abs(light): 
    ratio = abs(sound) / abs(light)
    light_sub = ratio * light
else:  
    light_sub = light
    
# Time evolution operator    
dt = 0.1
d_wave_sound = sound_wave(x, t + dt) - sound_wave(x, t) 
dx_sound = d_wave_sound / dt  

# Calculate subjective rate of change  
print(dx_sound)

import numpy as np

# Define constants  
c = 3e8 # speed of light (m/s)  
vs = 340 # speed of sound (m/s)
kB = 1.38e-23 # Boltzmann constant
ħ = 1.05e-34 # Reduced Planck's constant
T = 300 # Temperature (K) 

# Characteristic wavelengths  
λs = vs / 5000 # 500 Hz sound wave 
λγ = c / (4e14) # Near visible light

# Energies
Es = kB * T # Average phonon energy at T
Eγ = ħ * c / λγ # Photon energy  

# Subatomic size scale 
r0 = 1e-10 # 10 pm 

# Calculate key ratios
λ_ratio = λγ/λs # Wavelength ratio  
E_ratio = Eγ/Es # Energy ratio
size_ratio = λs/r0 # Phonon wavelength to subatomic scale

print(f'Wavelength Ratio: {λ_ratio:.5g}')
print(f'Energy Ratio: {E_ratio:.5g}') 
print(f'Phonon Size Ratio: {size_ratio:.2f}')

def forces(positions, velocities):
    # Placeholder function to calculate forces
    # This is just a dummy example and should be replaced with actual physics
    num_atoms = positions.shape[0]
    force_matrix = np.zeros((num_atoms, 3))  # Initialize a matrix to store forces
    
    # Example: Simple repulsive force from the origin
    for i in range(num_atoms):
        direction_to_origin = origin - positions[i]
        distance_to_origin = np.linalg.norm(direction_to_origin)
        if distance_to_origin > 0:  # Avoid division by zero
            force_magnitude = 1 / distance_to_origin**2  # Inverse square law as an example
            force_direction = direction_to_origin / distance_to_origin
            force_matrix[i] = force_magnitude * force_direction
    
    return force_matrix


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set parameters
num_atoms = 100         # Number of atoms
atom_mass = 39.95       # AMU - Argon 
box_size = 5            # Simulation box size (Angstroms)
dt = 0.001              # Simulation timestep (ps)
origin = [2.5, 2.5, 2.5] # Coordinate origin point (Angstroms)
temp = 5                # Temperature (K)

# Generate random atom positions and velocities  
positions = box_size * np.random.rand(num_atoms, 3)
velocities = stats.norm.rvs(loc=0, scale=np.sqrt(temp*1.38e-23/atom_mass), 
                            size=(num_atoms, 3))

def calculate_friction(positions, velocities):
    # Distance from origin
    delta_r = positions - origin
    
    # Radial component of velocity
    v_radial = np.sum(delta_r * velocities, axis=1) / np.linalg.norm(delta_r, axis=1)
    
    # Kinetic friction coefficient 
    mu_k = np.abs(v_radial / velocities[:,0])
    
    return np.average(mu_k)
    
# Calculate friction over time
mu_k = []
for i in range(100): 
    velocities += forces(positions, velocities) * dt 
    mu_k.append(calculate_friction(positions, velocities))
    
plt.plot(mu_k)
plt.xlabel('Timestep')
plt.ylabel('Friction Coeff')

import numpy as np

# Parameters
N_phonons = 100
box_size = 10  

# Randomly generate phonon properties
positions = np.random.rand(N_phonons, 3) * box_size  
scales = np.random.rand(N_phonons) * 0.5 + 0.1   
sizes = np.random.exponential(scale=3, size=N_phonons)

# Categorization functions 
def categorize_by_position(phonon):
    if phonon[0] < box_size/3:
        return "left"
    elif phonon[0] < 2*box_size/3:
        return "middle"
    else:
        return "right"

def categorize_by_scale(phonon):
    if phonon < 0.2:
        return "small"
    elif phonon < 0.4: 
        return "medium"
    else:
        return "large"
        
def categorize_by_size(phonon):
    if phonon < 2:
        return "small"
    elif phonon < 5:
        return "medium"
    else:
        return "large"
        
# Apply categorization
position_categories = [categorize_by_position(p) for p in positions] 
scale_categories = [categorize_by_scale(s) for s in scales]
size_categories = [categorize_by_size(s) for s in sizes]

print(position_categories)
print(scale_categories) 
print(size_categories)

import numpy as np
import matplotlib.pyplot as plt

# Phonon vertices
vertices = np.array([[0, 0], [1, 0], [0, 1]])

# Translation vectors
translations = np.array([[0.1, 0.2], 
                         [-0.1, 0.3],
                         [0.2, -0.1]])

# Time points
times = np.linspace(0, 10, 100)

# Initialize an array to hold translated vertices over time
translated = np.zeros((len(times), len(vertices), 2))

# Apply translations over time
for t_index, t in enumerate(times):
    for v_index, v in enumerate(vertices):
        translated[t_index, v_index] = v + translations[v_index]*t

# Plot vertex positions over time
for v_index in range(len(vertices)):
    x = translated[:, v_index, 0]
    y = translated[:, v_index, 1]
    
    plt.plot(times, x, label=f'Vertex {v_index+1} X')
    plt.plot(times, y, label=f'Vertex {v_index+1} Y')
    
plt.title("Phonon Vertex Translation Over Time")  
plt.legend()
plt.show()

import numpy as np
from scipy import integrate 
import matplotlib.pyplot as plt

# Phonon frequency (THz)
frequency = 5  

# Boltzmann constant 
kb = 1.38e-23 # Boltzmann constant 

# Calculate phonon occupation number at initial time
# Note: This initial calculation was incorrect in the context of dynamics; 
# it's provided here as a placeholder.
T_initial = 300  # Initial temperature (K)
exp_term_initial = np.exp(-frequency * kb * T_initial / (kb * T_initial))
n0 = 1 / (exp_term_initial - 1)

# Time-dependent temperature function
def T(t): 
    return 300 + 10*t  # Linearly increasing T

# Set up derivative to get rate of change of occupation number as a function of time
def occupation(n, t):
    # Calculate temperature at time t
    temp_at_t = T(t)
    # Calculate exponential term based on temperature at time t
    exp_term = np.exp(-frequency * kb * temp_at_t / (kb * temp_at_t))
    # Return the rate of change of occupation number
    return 1 / (exp_term - 1)  # This formula is incorrect for a rate of change and needs adjustment

# Integrate phonon occupation number over time
t_eval = np.linspace(0, 5, 100)
n_t = integrate.odeint(occupation, n0, t_eval)[:,0]

# Plot  
plt.plot(t_eval, n_t) 
plt.xlabel('Time (a.u.)')
plt.ylabel('Phonon Occupation Number')
plt.title('Phonon Occupation Over Time')
plt.show()

import numpy as np 
from sklearn import svm

# Generate some sample data
X = np.random.rand(100,2) 
y = np.random.randint(0,2,100)

# Train an SVM model
clf = svm.SVC()
clf.fit(X, y)

# Get the support vectors
support_vectors = clf.support_vectors_ 

# Calculate probability estimates 
# using distance from decision boundary
distances = clf.decision_function(X)  
probabilities = 1 / (1 + np.exp(-distances))

# Normalize probabilities to sum to 1
probabilities /= np.sum(probabilities)

# Sample random points with probability 
# proportional to probability estimate
sample_indices = np.random.choice(len(X), 10, p=probabilities)
samples = X[sample_indices]

print("Sampled Points:")
print(samples)
print("With Probability Estimates:")
print(probabilities[sample_indices])

import numpy as np
from sklearn import svm

# Generate sample data with mass feature 
num_samples = 500
X = np.random.rand(num_samples, 2) 
mass = np.random.exponential(scale=5, size=num_samples)
y = np.array([1 if m > 2 else 0 for m in mass])

# SVM with RBF kernel
clf = svm.SVC(kernel='rbf', gamma=0.1)  
clf.fit(X, y)  

# Get support vectors
support_vectors = clf.support_vectors_  

# Get mass values of support vectors  
vector_masses = mass[clf.support_]  

# Calculate mean, min, max of support vector mass values
vector_mass_mean = np.mean(vector_masses) 
vector_mass_min = np.min(vector_masses)
vector_mass_max = np.max(vector_masses)

print(f'Mean Support Vector Mass: {vector_mass_mean:.3f}') 
print(f'Min Support Vector Mass: {vector_mass_min:.3f}')
print(f'Max Support Vector Mass: {vector_mass_max:.3f}')

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate sample data with mass feature
num_samples = 500
X = np.random.rand(num_samples, 2)
mass = np.random.exponential(scale=10, size=num_samples)

# SVM with RBF kernel for regression
clf = SVR(kernel='rbf', gamma=0.5)
clf.fit(X, mass)

# Predict mass using the trained SVR model to determine support vectors
# Note: SVR does not have a `support_vectors_` attribute like SVC, but you can get the support
# vectors by indexing X with clf.support_
support_vectors = X[clf.support_]

# Assuming transformation logic remains the same, though it's typically not applied this way in regression
# Just for demonstration, let's proceed with your transformation logic
trans_vectors = []
vector_masses = mass[clf.support_]
mass_levels = [5, 10, 20] # Mass thresholds
for v, m in zip(support_vectors, vector_masses):
    if m < mass_levels[0]:
        # Lowest mass transformation
        v_new = v + 0.1
    elif m < mass_levels[1]:
        # Middle mass transformation
        v_new = v - 0.2
    else:
        # Highest mass transformation
        v_new = v / 2
    trans_vectors.append(v_new)

trans_vectors = np.array(trans_vectors)

# Plot transformed support vector unit locations
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], label='Original Vectors')
plt.scatter(trans_vectors[:, 0], trans_vectors[:, 1], label='Transformed Vectors')
plt.legend()
plt.title('Support Vector Transformations')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

# Generate data across dimensions 
dim1 = np.linspace(0, 10, 30)
dim2 = np.random.normal(loc=5, size=30) 
dim3 = np.random.uniform(low=0, high=10, size=30)
dims = np.vstack([dim1, dim2, dim3]).T

values = np.sin(dim1) + dim2 + np.random.normal(size=30)

# Train 3D linear regression 
lr = LinearRegression()
lr.fit(dims, values)

# Phonon properties    
phonon_locs = np.array([[2, 5, 3], 
                        [7, 3, 8],  
                        [4, 7, 5]])

# Plot regression and connections  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(phonon_locs[:, 0], phonon_locs[:, 1], phonon_locs[:, 2], 'go')

# Check if there are at least 4 points before constructing ConvexHull
if len(phonon_locs) >= 4:
    hull = ConvexHull(phonon_locs)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Loop back to the first point
        ax.plot(phonon_locs[s, 0], phonon_locs[s, 1], phonon_locs[s, 2], 'r-')
else:
    print("Not enough points to construct a ConvexHull in 3D.")

ax.scatter(dims[:, 0], dims[:, 1], dims[:, 2])
ax.plot_trisurf(dims[:, 0], dims[:, 1], lr.predict(dims), linewidth=0, alpha=0.5)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')

plt.title('Multi-Dimensional Regression and Phonon Connection')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

# Create 3D phonon density grid  
dim1 = np.linspace(0, 20, 40)
dim2 = np.linspace(0, 20, 40)[:, None] * np.ones(40)[None, :]
dim3 = np.linspace(0, 20, 40)
# Assuming you intended for z to be a 3D array; the original code only provided a 2D context.
# Let's create a 3D grid to use with marching_cubes.
Z = np.zeros((40, 40, 40))
for i in range(40):
    Z[i, :, :] = 5 * np.sin(np.sqrt(dim1[i]**2 + dim2**2))

# Set phonon density threshold for splice  
density_th = 2  

# Generate mesh  
verts, faces, normals, values = marching_cubes(volume=Z, level=density_th)  

# Extract x, y, z coordinates from verts for plotting
x, y, z = verts.T

# Plot splice  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(x, y, faces, z, cmap='viridis', lw=0.5)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Phonon Density')

plt.title('Phonon Matter Splice at Density = {:.3f}'.format(density_th))  
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
# Create 2D spaces
X1, Y1 = np.meshgrid(np.linspace(0,10,100), np.linspace(0,10,100))
Z1 = np.sin(X1)**2 + Y1/5

X2, Y2 = np.meshgrid(np.linspace(0,15,100), np.linspace(0,20,100)) 
Z2 = X2/3 - Y2**2  

# Phonon states 
phonons1 = 3 * np.sin(X1) * np.cos(Y1) 
phonons2 = 5 * np.exp(-((X2-5)**2 + (Y2-8)**2)/5)

# Fusion function 
def fuse(phonon1, phonon2):
    return np.sqrt(phonon1**2 + phonon2**2)

# Prepare interpolation grids more explicitly
interp1 = interpolate.RectBivariateSpline(np.linspace(0,10,100), np.linspace(0,10,100), phonons1)  
interp2 = interpolate.RectBivariateSpline(np.linspace(0,15,100), np.linspace(0,20,100), phonons2)

# Create a mesh for evaluation
X, Y = np.meshgrid(np.linspace(0, 15, 100), np.linspace(0, 20, 100))
phonon_interp1 = interp1.ev(X, Y)
phonon_interp2 = interp2.ev(X, Y)

phonon_fused = fuse(phonon_interp1, phonon_interp2)

# Plot fused spacetime
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
X_flat, Y_flat = np.meshgrid(np.linspace(0, 15, 100), np.linspace(0, 20, 100))
ax.plot_surface(X_flat, Y_flat, phonon_fused, cmap='viridis')
ax.set_title('Fused Phonon State')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fused Phonon Density')

plt.show()

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Create 2D spaces
x1 = np.linspace(0, 5, 30)
y1 = np.linspace(0, 5, 30)  
X1, Y1 = np.meshgrid(x1, y1)
Z1 = np.sin(X1) + Y1  

x2 = np.linspace(0, 10, 30)
y2 = np.linspace(0, 10, 30)
X2, Y2 = np.meshgrid(x2, y2) 
Z2 = X2**2 - Y2

# Convert verts1 and verts2 to numpy arrays right after their definition
verts1 = np.array([[0, 1, 0], [3, 2, 0.8], [4, 4, 0.5]])
verts2 = np.array([[0, 0, 0], [7, 5, 4], [9, 8, 2]])

# Now your existing transformations and translations can proceed unchanged
# Translation vectors
trans_v = [np.array([1, 0.5, 2]), np.array([0, -1, 0.1])]

# Translate vertices in each space
verts1_trans = []
verts2_trans = []

for v in verts1:
    verts1_trans.append(v + trans_v[0])

for v in verts2:
    verts2_trans.append(v + trans_v[1])

verts1_trans = np.array(verts1_trans)
verts2_trans = np.array(verts2_trans)

# Plotting code remains unchanged and should now work without the TypeError
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use verts1_trans and verts2_trans for plotting as they are the translated versions
ax.scatter(verts1_trans[:, 0], verts1_trans[:, 1], verts1_trans[:, 2], c='r')
ax.scatter(verts2_trans[:, 0], verts2_trans[:, 1], verts2_trans[:, 2], c='g')

for v1, v2 in zip(verts1_trans, verts2_trans):
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], c='k')



# Note: You may need to adjust the plotting for Z_fused as it seems to be intended to be plotted over X1, Y1, Z1 or similar
# This example does not correct logic for Z_fused as it's unclear without more context

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title("Fused Cross-Dimensional Vertices")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define vectors and steps
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 2.0, 0.0])
v3 = np.array([0.0, 0.0, 3.0])
step = 0.1  
h_feedback = np.array([0.1, 0, 0])

# Unity constraint function
unity_constraint = lambda v: v / np.linalg.norm(v)

# Apply translations and feedback
trans_v1 = unity_constraint(v1 + h_feedback)
trans_v2 = unity_constraint(v2 + h_feedback)
trans_v3 = unity_constraint(v3 + h_feedback)

# Phonon field density calculation (simplified for demonstration)
X, Y, Z = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 10, 30), np.linspace(0, 5, 30))
phonon_density = np.sin(X + Y + Z) > 0  # Example condition for voxels

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Quiver for translated vectors
ax.quiver(0, 0, 0, trans_v1[0], trans_v1[1], trans_v1[2], color='r')
ax.quiver(0, 0, 0, trans_v2[0], trans_v2[1], trans_v2[2], color='g')
ax.quiver(0, 0, 0, trans_v3[0], trans_v3[1], trans_v3[2], color='b')

# Voxels for phonon density
# Assuming phonon_density is a (30, 30, 30) array indicating occupancy
ax.voxels(phonon_density, facecolors='yellow', edgecolors='k')


# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Vector Translation with Horizontal Feedback')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vector translations
# Vector translations initialized as float64 explicitly
v1 = np.array([1, 0, 0], dtype=np.float64) 
v2 = np.array([0, 2, 0], dtype=np.float64)
v3 = np.array([0, 0, 3], dtype=np.float64)

# Translation amount each step
step = 0.1  

# Phonon field density
x = np.linspace(0, 10, 30)
y = np.linspace(0, 10, 30)[:, None]
z = np.ones(30)[:, None] * 5
X, Y, Z = np.meshgrid(x, y, z)
phonon_density = np.sin(X+Y+Z)

# Unity constraint 
unity_constraint = lambda v : v / np.linalg.norm(v)

# Horizontal feedback  
h_feedback = np.array([0.1, 0, 0])

# Initialize vertical translations  
trans_v1 = v1 
trans_v2 = v2
trans_v3 = v3   

# Iterate translation and feedback 
for i in range(100):
    
    trans_v1 += step * v1  
    trans_v2 += step * v2
    trans_v3 += step * v3
    
    
    trans_v1 = unity_constraint(trans_v1) 
    trans_v2 = unity_constraint(trans_v2)
    trans_v3 = unity_constraint(trans_v3)  
    
    trans_v1 += h_feedback 
    trans_v2 += h_feedback
    trans_v3 += h_feedback
    
# Plot    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, trans_v1[0], trans_v1[1], trans_v1[2], color='r')
ax.quiver(0, 0, 0, trans_v2[0], trans_v2[1], trans_v2[2], color='g') 
ax.quiver(0, 0, 0, trans_v3[0], trans_v3[1], trans_v3[2], color='b')

ax.voxels(phonon_density, facecolors='y', edgecolors='k')

ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')

plt.title('Vector Translation with Horizontal Feedback')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define vertical translation and horizontal feedback
h_vec = np.array([0.1, 0, 0])
v_vec = np.array([0, 0.1, 0])

# Create slicing coordinates
x = np.linspace(0, 1, 20) 
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)

# Initialize translation vectors 
trans_v = v_vec
trans_h = h_vec

# Store transformation slices
trans_splice = []  
for i in range(20):
    for j in range(20):
        
        # Translation step 
        trans_v += v_vec  
        trans_h += h_vec 
        
        # Append to slice storage
        trans_splice.append([trans_v[0], trans_v[1], trans_h[0], trans_h[1]])
        
    # Reset horizontal translation        
    trans_h = h_vec
    
trans_splice = np.array(trans_splice)

# Reshape into grid
splice = np.hsplit(trans_splice.reshape(20,20,4), 2)
VT = splice[0].reshape(20,20,2)
HT = splice[1].reshape(20,20,2)

# Plot splice  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, VT[:,:,0], color='y')  
ax.plot_surface(X, Y+0.1, HT[:,:,0], color='b')

ax.set_title('Transformation Splice')
plt.show()