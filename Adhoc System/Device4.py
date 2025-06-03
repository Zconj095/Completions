import time

class Device:
    def __init__(self, name, position, range):
        self.name = name
        self.position = position
        self.range = range
        self.connected_devices = []
        self.allowed_devices = []
        self.fixed_delay = 0.05  # Fixed delay in seconds (50 milliseconds)

    def can_connect(self, other_device):
        # ... existing can_connect logic ...

    def send_message(self, other_device, message):
        if self.can_connect(other_device):
            time.sleep(self.fixed_delay)  # Simulate the fixed delay
            print(f"Message from {self.name} to {other_device.name}: '{message}' after {self.fixed_delay * 1000:.0f} milliseconds")
        else:
            print(f"{self.name} cannot connect with {other_device.name} due to barriers")

# Example usage
device_a = Device("Device A", (0, 0), 50)
device_b = Device("Device B", (30, 40), 50)

device_a.allowed_devices = ["Device B"]
device_b.allowed_devices = ["Device A"]

device_a.send_message(device_b, "Hello Device B")
