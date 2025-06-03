import json
import glob
import matplotlib.pyplot as plt

def visualize_latest_analysis():
    """Visualize the latest analysis result."""
    list_of_files = glob.glob('analysis_result_*.json')
    latest_file = max(list_of_files, key=lambda x: x.split('_')[2])
    
    with open(latest_file, 'r') as file:
        analysis_result = json.load(file)
    
    # Simulate visualization: Display health status
    status = analysis_result['health_status']
    plt.figure(figsize=(5, 3))
    plt.text(0.5, 0.5, status, fontsize=12, ha='center')
    plt.axis('off')
    plt.title("Health Status")
    plt.show()

if __name__ == "__main__":
    visualize_latest_analysis()
