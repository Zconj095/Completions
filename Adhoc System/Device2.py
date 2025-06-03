import time
import random

class Device:
    def __init__(self, name, position, range, max_speed):
        self.name = name
        self.position = position
        self.range = range
        self.max_speed = max_speed  # Maximum data transfer speed in Mbps
        self.connected_devices = []
        self.allowed_devices = []

    def can_connect(self, other_device):
        # ... existing can_connect logic ...

    def calculate_speed(self, other_device):
        """ Calculate actual data transfer speed based on network condition. """
        base_delay = self.calculate_delay(other_device)
        # Simulating network condition effect on speed (e.g., weaker signal = slower speed)
        speed = self.max_speed * (1 - base_delay)  # Example formula
        return max(speed, 0.1)  # Minimum speed to ensure some data transfer

    def calculate_delay(self, other_device):
        """ Calculate delay based on distance and other factors. """
        # ... existing calculate_delay logic ...

    def transfer_data(self, other_device, data_size):
        """ Simulate data transfer, considering network conditions. """
        if self.can_connect(other_device):
            speed = self.calculate_speed(other_device)  # Mbps
            transfer_time = data_size / speed  # Data size in MB, transfer time in seconds
            time.sleep(transfer_time)  # Simulating the time taken for transfer
            print(f"Transferred {data_size} MB from {self.name} to {other_device.name} at {speed:.2f} Mbps, taking {transfer_time:.2f} seconds")
        else:
            print(f"{self.name} cannot transfer data to {other_device.name} due to barriers or range.")

# Example usage
device_a = Device("Device A", (0, 0), 50, max_speed=10)  # 10 Mbps max speed
device_b = Device("Device B", (30, 40), 50, max_speed=10)

device_a.allowed_devices = ["Device B"]
device_b.allowed_devices = ["Device A"]

# Simulating data transfer
data_size = 5  # Data size in MB
device_a.transfer_data(device_b, data_size)
