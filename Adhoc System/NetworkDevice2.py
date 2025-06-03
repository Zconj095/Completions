import random

class NetworkDevice:
    def __init__(self, name, connection_type, is_edge_device=False, position=(0, 0), range=50):
        self.name = name
        self.connection_type = connection_type  # 'Ethernet' or 'Wireless'
        self.is_edge_device = is_edge_device  # Indicates if the device is an edge computing device
        self.position = position
        self.range = range
        self.connected_devices = []

    def process_data(self, data):
        """ Simulate data processing in edge device. """
        if self.is_edge_device:
            processed_data = f"Processed {data} in {self.name}"
            print(processed_data)
            return processed_data
        else:
            print(f"{self.name} is not an edge computing device.")
            return None

    def transfer_data(self, other_device, data):
        """ Simulate data transfer considering offline capabilities. """
        if self.is_in_range(other_device):
            print(f"Transferring data from {self.name} to {other_device.name}...")
            processed_data = other_device.process_data(data)
            return processed_data
        else:
            print(f"{other_device.name} is out of range for {self.name}.")
            return None

    def is_in_range(self, other_device):
        if self.connection_type == 'Wireless':
            distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                        (self.position[1] - other_device.position[1]) ** 2) ** 0.5
            return distance <= self.range
        else:
            return True  # Wired connections are not limited by range

    def switch_connection(self, new_connection_type):
        """ Switch between Wired and Wireless connection. """
        self.connection_type = new_connection_type
        print(f"{self.name} switched to {new_connection_type} connection.")

# Example usage
edge_device = NetworkDevice("Edge Device", "Wireless", True, (0, 0), 50)
normal_device = NetworkDevice("Normal Device", "Wireless", False, (30, 40), 50)

# Simulating data processing and transfer
data = "Sample Data"
edge_device.transfer_data(normal_device, data)

# Switching connection type
edge_device.switch_connection("Ethernet")
