class NetworkDevice:
    def __init__(self, protocol, wavelength, signal_strength, noise_level):
        self.protocol = protocol
        self.wavelength = wavelength
        self.signal_strength = signal_strength
        self.noise_level = noise_level
        self.transfer_rate = self.calculate_transfer_rate()

    def calculate_transfer_rate(self):
        # Define standard transfer rates for protocols (in Mbps)
        protocol_rates = {'WiFi': 100, 'Bluetooth': 2, 'LTE': 50}
        base_rate = protocol_rates.get(self.protocol, 1)

        # Adjust rate based on wavelength (hypothetical adjustment)
        wavelength_factor = 1 if 400 <= self.wavelength <= 700 else 0.5

        # Adjust rate based on signal strength and noise level
        signal_adjustment = self.signal_strength / (self.noise_level + 1)

        # Calculate final transfer rate
        return base_rate * wavelength_factor * signal_adjustment

    def transfer_data(self, data_size):
        # Simulate data transfer
        transfer_time = (data_size * 8) / self.transfer_rate  # Convert KB to Kb
        print(f"Transferring {data_size} KB at {self.transfer_rate:.2f} Mbps takes {transfer_time:.2f} seconds.")

# Example usage
device = NetworkDevice(protocol='WiFi', wavelength=500, signal_strength=80, noise_level=20)
device.transfer_data(1000)  # Transfer 1000 KB of data
