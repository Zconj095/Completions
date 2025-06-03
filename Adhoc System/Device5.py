import time
import random

class Device:
    def __init__(self, name, position, range):
        self.name = name
        self.position = position
        self.range = range
        self.connected_devices = []
        self.allowed_devices = []

    def can_connect(self, other_device):
        # ... existing can_connect logic ...

    def calculate_delay(self, other_device):
        """ Calculate communication delay based on various factors. """
        # Example: Basic delay based on distance
        distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                    (self.position[1] - other_device.position[1]) ** 2) ** 0.5
        delay = distance / 10  # Example formula for delay calculation

        # Add random additional delay to simulate network conditions
        additional_delay = random.uniform(0, 1)  # Random delay between 0 to 1 second
        return delay + additional_delay

    def send_message(self, other_device, message):
        if self.can_connect(other_device):
            delay = self.calculate_delay(other_device)
            time.sleep(delay)  # Simulate the delay
            print(f"Message from {self.name} to {other_device.name}: '{message}' after {delay:.2f} seconds delay")
        else:
            print(f"{self.name} cannot connect with {other_device.name} due to barriers")

# Example usage
device_a = Device("Device A", (0, 0), 50)
device_b = Device("Device B", (30, 40), 50)

device_a.allowed_devices = ["Device B"]
device_b.allowed_devices = ["Device A"]

device_a.send_message(device_b, "Hello Device B")
