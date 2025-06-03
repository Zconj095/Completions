import time

class Device:
    def __init__(self, name, position, range):
        self.name = name
        self.position = position
        self.range = range
        self.connected_devices = []
        self.allowed_devices = []
        self.fixed_delay = 0.05  # 50 milliseconds delay

    def can_connect(self, other_device):
        # ... existing can_connect logic ...

    def upload_data(self, other_device, data):
        """ Simulate uploading data to another device. """
        if self.can_connect(other_device):
            time.sleep(self.fixed_delay)  # Simulate network delay
            other_device.download_data(self, data)
            print(f"{self.name} uploaded data to {other_device.name}")

    def download_data(self, from_device, data):
        """ Simulate downloading data from another device. """
        print(f"{self.name} received data from {from_device.name}: {data}")

# Example usage
device_a = Device("Device A", (0, 0), 50)
device_b = Device("Device B", (30, 40), 50)

device_a.allowed_devices = ["Device B"]
device_b.allowed_devices = ["Device A"]

# Simulating data transfer
data_to_transfer = "Sample Data"
device_a.upload_data(device_b, data_to_transfer)
