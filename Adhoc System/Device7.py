class Device:
    def __init__(self, name, position, range):
        self.name = name
        self.position = position
        self.range = range
        self.connected_devices = []

    def can_connect(self, other_device):
        distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                    (self.position[1] - other_device.position[1]) ** 2) ** 0.5
        return distance <= self.range

    def connect(self, other_device):
        if self.can_connect(other_device):
            self.connected_devices.append(other_device.name)
            other_device.connected_devices.append(self.name)
            print(f"{self.name} connected with {other_device.name}")
        else:
            print(f"{self.name} cannot connect with {other_device.name} due to distance")

# Example devices
device_a = Device("Device A", (0, 0), 50)  # Device A at position (0,0) with a range of 50
device_b = Device("Device B", (30, 40), 50) # Device B at position (30,40) with a range of 50
device_c = Device("Device C", (100, 100), 50) # Device C at position (100,100), out of range

# Attempting connections
device_a.connect(device_b)  # Should succeed
device_a.connect(device_c)  # Should fail due to distance
