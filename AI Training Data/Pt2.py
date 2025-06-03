import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import lag_plot, autocorrelation_plot
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data():
    """Load and merge transistor count data"""
    # Load CPU data
    df = pd.read_csv('transcount.csv')
    df = df.groupby('year').aggregate(np.mean)
    
    # Load GPU data
    gpu = pd.read_csv('gpu_transcount.csv')
    gpu = gpu.groupby('year').aggregate(np.mean)
    
    # Merge datasets
    merged_df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
    merged_df = merged_df.replace(np.nan, 0)
    
    return merged_df, gpu

def plot_moore_law_trend(df, gpu):
    """Plot Moore's Law trend with polynomial fit"""
    years = df.index.values
    counts = df['trans_count'].values
    gpu_counts = df['gpu_trans_count'].values
    
    # Remove zeros for log calculation
    valid_counts = counts[counts > 0]
    valid_years = years[counts > 0]
    
    # Polynomial fit
    poly = np.polyfit(valid_years, np.log(valid_counts), deg=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Semilog plot
    ax1.semilogy(valid_years, valid_counts, 'o', label='CPU Data', markersize=8)
    ax1.semilogy(valid_years, np.exp(np.polyval(poly, valid_years)), 
                 '--', label=f'Exponential Fit (slope: {poly[0]:.3f})', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Transistor Count')
    ax1.set_title('Moore\'s Law - Exponential Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot with GPU data
    cnt_log = np.log(counts[counts > 0])
    gpu_valid = gpu_counts[counts > 0]
    
    scatter = ax2.scatter(valid_years, cnt_log, 
                         c=valid_years, 
                         s=20 + 200 * gpu_valid/gpu_valid.max() if gpu_valid.max() > 0 else 20,
                         alpha=0.7, cmap='viridis')
    ax2.plot(valid_years, np.polyval(poly, valid_years), 'r--', 
             label='Trend Line', linewidth=2)
    
    # Annotate first GPU
    if len(gpu[gpu['gpu_trans_count'] > 0]) > 0:
        gpu_start = gpu[gpu['gpu_trans_count'] > 0].index.min()
        if gpu_start in df.index:
            y_ann = np.log(df.at[gpu_start, 'trans_count'])
            ax2.annotate(f'First GPU\n{gpu_start}', 
                        xy=(gpu_start, y_ann), 
                        arrowprops=dict(arrowstyle="->", color='red'),
                        xytext=(-30, +70), textcoords='offset points',
                        fontsize=10, ha='center')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Log Transistor Count')
    ax2.set_title('Moore\'s Law - Log Scale with GPU Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax2, label='Year')
    plt.tight_layout()
    plt.show()

def plot_3d_surface(df):
    """Create 3D surface plot"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    cpu_data = df['trans_count'].values
    gpu_data = df['gpu_trans_count'].values
    years = df.index.values
    
    # Filter valid data
    valid_mask = (cpu_data > 0) & (gpu_data > 0)
    
    if np.sum(valid_mask) > 0:
        x_valid = years[valid_mask]
        y_valid = np.log(cpu_data[valid_mask])
        z_valid = np.log(gpu_data[valid_mask])
        
        # Create meshgrid for surface
        xi = np.linspace(x_valid.min(), x_valid.max(), 20)
        yi = np.linspace(y_valid.min(), y_valid.max(), 20)
        X, Y = np.meshgrid(xi, yi)
        
        # Simple interpolation for Z
        Z = np.zeros_like(X)
        for i in range(len(xi)):
            for j in range(len(yi)):
                distances = np.sqrt((x_valid - xi[i])**2 + (y_valid - yi[j])**2)
                if len(distances) > 0:
                    closest_idx = np.argmin(distances)
                    Z[j, i] = z_valid[closest_idx]
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.scatter(x_valid, y_valid, z_valid, c='red', s=50)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Log CPU Transistor Count')
        ax.set_zlabel('Log GPU Transistor Count')
        ax.set_title('Moore\'s Law - 3D Relationship')
        
        plt.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

def plot_comparative_analysis(df):
    """Create comprehensive comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time series plot
    df_plot = df[df > 0].replace(0, np.nan)
    df_plot.plot(logy=True, ax=ax1)
    ax1.set_title('Transistor Count Over Time (Log Scale)')
    ax1.set_ylabel('Transistor Count (Log)')
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot CPU vs GPU
    valid_data = df[(df['trans_count'] > 0) & (df['gpu_trans_count'] > 0)]
    if len(valid_data) > 0:
        valid_data.plot(kind='scatter', x='trans_count', y='gpu_trans_count', 
                       loglog=True, ax=ax2, alpha=0.7, s=60)
        ax2.set_title('CPU vs GPU Transistor Count')
        ax2.grid(True, alpha=0.3)
    
    # Lag plot
    cpu_log = np.log(df['trans_count'][df['trans_count'] > 0])
    if len(cpu_log) > 1:
        lag_plot(cpu_log, ax=ax3)
        ax3.set_title('Lag Plot - CPU Transistor Count')
        ax3.grid(True, alpha=0.3)
    
    # Autocorrelation plot
    if len(cpu_log) > 1:
        autocorrelation_plot(cpu_log, ax=ax4)
        ax4.set_title('Autocorrelation - CPU Transistor Count')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_growth_metrics(df):
    """Calculate and display growth metrics"""
    cpu_data = df['trans_count'][df['trans_count'] > 0]
    years = cpu_data.index.values
    counts = cpu_data.values
    
    if len(counts) > 1:
        # Calculate doubling time
        poly = np.polyfit(years, np.log(counts), deg=1)
        doubling_time = np.log(2) / poly[0]
        
        # Annual growth rate
        growth_rate = (np.exp(poly[0]) - 1) * 100
        
        print(f"Moore's Law Analysis:")
        print(f"Annual growth rate: {growth_rate:.1f}%")
        print(f"Doubling time: {doubling_time:.1f} years")
        print(f"R² correlation: {np.corrcoef(years, np.log(counts))[0,1]**2:.3f}")

def main():
    """Main execution function"""
    try:
        # Load and process data
        df, gpu = load_and_process_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Years covered: {df.index.min()} - {df.index.max()}")
        
        # Generate all plots
        plot_moore_law_trend(df, gpu)
        plot_3d_surface(df)
        plot_comparative_analysis(df)
        
        # Calculate metrics
        calculate_growth_metrics(df)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
