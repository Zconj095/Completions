import time
import random

class NetworkDevice:
    def __init__(self, name, position, range, protocol, wavelength, signal_strength, noise_level):
        self.name = name
        self.position = position
        self.range = range
        self.protocol = protocol
        self.wavelength = wavelength
        self.signal_strength = signal_strength
        self.noise_level = noise_level
        self.allowed_devices = []
        self.packet_delay = 0.0012  # 1.2 milliseconds delay
        self.packet_loss_rate = self.calculate_packet_loss_rate()
        self.transfer_rate = self.calculate_transfer_rate()  # Mbps

    def can_connect(self, other_device):
        distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                    (self.position[1] - other_device.position[1]) ** 2) ** 0.5
        return distance <= self.range

    def calculate_transfer_rate(self):
        # ... existing calculate_transfer_rate logic ...

    def calculate_packet_loss_rate(self):
        # ... existing packet loss calculation logic ...

    def transfer_data(self, other_device, data_size):
        if self.can_connect(other_device):
            packet_size = 10  # Assume packet size in KB
            total_packets = data_size / packet_size
            successful_packets = total_packets * (1 - self.packet_loss_rate)
            total_time = (data_size * 8) / self.transfer_rate  # Convert KB to Kb
            time.sleep(total_time)  # Simulating the time taken for transfer
            print(f"{self.name} transferred {data_size} KB to {other_device.name} with {successful_packets:.0f} successful packets in {total_time:.2f} seconds.")
        else:
            print(f"{self.name} cannot transfer data to {other_device.name} due to range or barriers.")

# Example usage
device_a = NetworkDevice("Device A", (0, 0), 50, 'WiFi', 500, 80, 20)
device_b = NetworkDevice("Device B", (30, 40), 50, 'WiFi', 500, 80, 20)

device_a.allowed_devices = ["Device B"]
device_b.allowed_devices = ["Device A"]

device_a.transfer_data(device_b, 1000)  # Transfer 1000 KB of data
