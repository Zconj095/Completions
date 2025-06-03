import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import ephem

# Define the Microbiome class since it's referenced but not defined
class Microbiome:
    def __init__(self, taxa_abundances):
        self.taxa_abundances = taxa_abundances
    
    def update_state(self, moon_phase, solar_cycle_phase):
        # Simulate microbiome response to celestial events
        updated_abundances = {}
        for taxon, abundance in self.taxa_abundances.items():
            multiplier = 1.0
            if moon_phase in ["Full Moon", "New Moon"]:
                multiplier *= 1.1
            if solar_cycle_phase == "Ascending Phase":
                multiplier *= 1.05
            updated_abundances[taxon] = abundance * multiplier
        return updated_abundances
    
    def signaling_cascade(self, moon_phase):
        # Calculate diversity response based on moon phase
        base_diversity = sum(self.taxa_abundances.values())
        if moon_phase in ["Full Moon", "New Moon"]:
            return base_diversity * 1.15
        return base_diversity

class Neuron:
    def __init__(self, voltage=-70): 
        self.voltage = voltage
        
    def update(self, current, moon_phase):
        if moon_phase in ["Full Moon", "New Moon"]:
            self.voltage += 0.7 * current  # Enhanced response
        else:
            self.voltage += 0.5 * current  # Normal response

# Global data structures
taxa_abundances = {
    "Bacteroides": 0.23,
    "Firmicutes": 0.27,
    "Actinobacteria": 0.12,
    "Proteobacteria": 0.18,
    "Fusobacteria": 0.05,
    "Other": 0.15
}

neuron_voltages = []

def get_moon_phase(date):
    """Calculate moon phase for a given date"""
    try:
        observer = ephem.Observer()
        observer.date = date.strftime("%Y/%m/%d")
        moon = ephem.Moon(observer)
        moon_phase_number = moon.phase / 100

        if moon_phase_number <= 0.05:
            return "New Moon"
        elif 0.05 < moon_phase_number <= 0.25:
            return "Waxing Crescent"
        elif 0.25 < moon_phase_number <= 0.45:
            return "First Quarter"
        elif 0.45 < moon_phase_number <= 0.55:
            return "Waxing Gibbous"
        elif 0.55 < moon_phase_number <= 0.95:
            return "Full Moon"
        elif 0.95 < moon_phase_number <= 0.75:
            return "Waning Gibbous"
        elif 0.75 < moon_phase_number <= 0.55:
            return "Last Quarter"
        else:
            return "Waning Crescent"
    except Exception as e:
        print(f"Error calculating moon phase: {e}")
        return "Unknown"

def get_solar_cycle_phase(date):
    """Calculate solar cycle phase for a given date"""
    cycle_length = 11
    reference_year = 2009
    
    year_difference = date.year - reference_year
    cycle_phase = (year_difference % cycle_length) / cycle_length
    
    return "Ascending Phase" if cycle_phase < 0.5 else "Descending Phase"

def get_user_input_date():
    """Get valid date input from user"""
    while True:
        input_date = input("Enter a date for simulation (YYYY-MM-DD): ")
        try:
            return datetime.strptime(input_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

def get_user_simulation_choice():
    """Get simulation type choice from user"""
    print("\nSelect a simulation type:")
    print("1. Neuron Activity")
    print("2. Microbiome Diversity")
    print("3. Both simulations")
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

def neuron_simulation(date, moon_phase, current=5):
    """Simulate neuron activity"""
    neuron = Neuron()
    neuron.update(current, moon_phase)
    neuron_voltages.append(neuron.voltage)
    
    print(f"\n--- Neuron Simulation Results ---")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"Moon Phase: {moon_phase}")
    print(f"Neuron Voltage: {neuron.voltage:.2f} mV")
    
    return neuron.voltage

def microbiome_simulation(date, moon_phase, solar_cycle_phase):
    """Simulate microbiome diversity"""
    microbiome = Microbiome(taxa_abundances)
    updated_abundances = microbiome.update_state(moon_phase, solar_cycle_phase)
    diversity_response = microbiome.signaling_cascade(moon_phase)
    
    print(f"\n--- Microbiome Simulation Results ---")
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"Moon Phase: {moon_phase}")
    print(f"Solar Cycle Phase: {solar_cycle_phase}")
    print(f"Diversity Response: {diversity_response:.3f}")
    print("Updated Abundances:")
    for taxon, abundance in updated_abundances.items():
        print(f"  {taxon}: {abundance:.3f}")
    
    return updated_abundances, diversity_response

def simulate_biology(choice, date, moon_phase, solar_cycle_phase):
    """Main simulation function"""
    results = {}
    
    if choice in ['1', '3']:
        results['neuron'] = neuron_simulation(date, moon_phase)
    
    if choice in ['2', '3']:
        abundances, diversity = microbiome_simulation(date, moon_phase, solar_cycle_phase)
        results['microbiome'] = {'abundances': abundances, 'diversity': diversity}
    
    return results

def visualize_data(data, title, x_label="Time", y_label="Value"):
    """Create visualization of simulation data"""
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linewidth=2, markersize=6)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_results(results):
    """Analyze simulation results"""
    if 'neuron' in results:
        print(f"\n--- Neuron Analysis ---")
        print(f"Current voltage: {results['neuron']:.2f} mV")
        
    if 'microbiome' in results:
        print(f"\n--- Microbiome Analysis ---")
        diversity = results['microbiome']['diversity']
        abundances = results['microbiome']['abundances']
        
        print(f"Total diversity index: {diversity:.3f}")
        print(f"Most abundant taxon: {max(abundances, key=abundances.get)}")
        print(f"Least abundant taxon: {min(abundances, key=abundances.get)}")

def display_help():
    """Display help information"""
    help_text = """
    ================================
    Biological and Celestial Event Simulation
    ================================
    
    This simulation models how celestial events (moon phases and solar cycles) 
    affect biological systems (neurons and microbiomes).
    
    Instructions:
    1. Choose simulation type: Neuron Activity, Microbiome Diversity, or Both
    2. Enter a date in YYYY-MM-DD format
    3. View results and analysis
    4. Optionally visualize data trends
    
    Features:
    - Real-time celestial calculations using PyEphem
    - Statistical analysis of results
    - Data visualization with matplotlib
    - Interactive parameter adjustment
    
    Tips:
    - Full Moon and New Moon phases have enhanced biological effects
    - Solar cycle phases influence microbiome diversity
    - Use visualization to track trends over multiple runs
    """
    print(help_text)

def run_simulation():
    """Main simulation runner"""
    print("=" * 50)
    print("BIOLOGICAL AND CELESTIAL SIMULATION")
    print("=" * 50)
    
    print("\nType 'help' for instructions or press Enter to continue:")
    if input().lower() == 'help':
        display_help()
    
    try:
        # Get user inputs
        date = get_user_input_date()
        simulation_choice = get_user_simulation_choice()
        
        # Calculate celestial events
        moon_phase = get_moon_phase(date)
        solar_cycle_phase = get_solar_cycle_phase(date)
        
        print(f"\n--- Celestial Conditions ---")
        print(f"Date: {date.strftime('%Y-%m-%d')}")
        print(f"Moon Phase: {moon_phase}")
        print(f"Solar Cycle Phase: {solar_cycle_phase}")
        
        # Run simulation
        results = simulate_biology(simulation_choice, date, moon_phase, solar_cycle_phase)
        
        # Analyze results
        analyze_results(results)
        
        # Optional visualization
        if neuron_voltages and input("\nVisualize neuron data? (y/n): ").lower() == 'y':
            visualize_data(neuron_voltages, "Neuron Voltage Over Time", "Simulation Run", "Voltage (mV)")
        
        return results
        
    except Exception as e:
        print(f"Simulation error: {e}")
        return None

def run_multiple_scenarios():
    """Run multiple simulation scenarios"""
    try:
        number_of_scenarios = int(input("How many scenarios would you like to simulate? "))
        all_results = []
        
        for i in range(number_of_scenarios):
            print(f"\n{'='*20} SCENARIO {i+1} {'='*20}")
            result = run_simulation()
            if result:
                all_results.append(result)
        
        print(f"\nCompleted {len(all_results)} successful simulations.")
        
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"Error running multiple scenarios: {e}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Run single simulation")
        print("2. Run multiple scenarios")
        print("3. Display help")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == '1':
            run_simulation()
        elif choice == '2':
            run_multiple_scenarios()
        elif choice == '3':
            display_help()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
