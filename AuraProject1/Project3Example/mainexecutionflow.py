from realtimefeedbacksimulation import *
from SimulatedDataCollection import *
from basicdataprocessing import *
def main():
    for _ in range(5):  # Simulate a series of data collection and processing iterations
        aura_data = simulate_aura_data_collection()
        processed_data = process_aura_data(aura_data)
        display_real_time_feedback(processed_data)
        time.sleep(2)  # Simulate a delay between data collections

if __name__ == "__main__":
    main()
