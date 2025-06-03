class Device:
    def __init__(self, name, position, range):
        self.name = name
        self.position = position
        self.range = range
        self.connected_devices = []
        self.allowed_devices = []  # List of devices allowed to connect

    def can_connect(self, other_device):
        # Check distance
        distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                    (self.position[1] - other_device.position[1]) ** 2) ** 0.5
        if distance > self.range:
            return False

        # Check if the device is allowed (software barrier)
        if other_device.name not in self.allowed_devices:
            return False

        return True

    def connect(self, other_device):
        if self.can_connect(other_device):
            self.connected_devices.append(other_device.name)
            other_device.connected_devices.append(self.name)
            print(f"{self.name} connected with {other_device.name}")
        else:
            print(f"{self.name} cannot connect with {other_device.name} due to barriers")

# Example devices
device_a = Device("Device A", (0, 0), 50)
device_b = Device("Device B", (30, 40), 50)
device_c = Device("Device C", (10, 10), 50)

# Adding allowed devices (software barrier)
device_a.allowed_devices = ["Device B"]  # Only Device B can connect with Device A
device_b.allowed_devices = ["Device A", "Device C"]  # Device B can connect with A and C
device_c.allowed_devices = ["Device B"]  # Only Device B can connect with Device C

# Attempting connections
device_a.connect(device_b)  # Should succeed
device_a.connect(device_c)  # Should fail due to software barrier
device_b.connect(device_c)  # Should succeed
