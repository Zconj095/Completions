import time

class Device:
    def __init__(self, name, position, range):
        self.name = name
        self.position = position
        self.range = range
        self.connected_devices = []
        self.allowed_devices = []
        self.transfer_rate = 15  # Fixed transfer rate in Kbps
        self.packet_delay = 0.0012  # 1.2 milliseconds delay

    def can_connect(self, other_device):
        # ... existing can_connect logic ...

    def transfer_data(self, other_device, data_size):
        """ Simulate data transfer with variable packet sizes. """
        if self.can_connect(other_device):
            # Define thresholds for data sizes (in KB)
            large_data_threshold = 1024  # 1 MB
            medium_data_threshold = 512  # 512 KB

            if data_size > large_data_threshold:
                packet_size = 100  # Larger packets for large data (in KB)
            elif data_size > medium_data_threshold:
                packet_size = 50   # Medium packets for medium data
            else:
                packet_size = 10   # Smaller packets for small data

            total_packets = data_size / packet_size
            total_delay = total_packets * self.packet_delay  # Total delay for all packets
            total_time = (data_size * 8) / self.transfer_rate + total_delay  # Total transfer time, data_size * 8 to convert KB to Kb

            time.sleep(total_time)  # Simulating the time taken for transfer
            print(f"Transferred {data_size} KB from {self.name} to {other_device.name} taking {total_time:.2f} seconds with {total_packets:.0f} packets")
        else:
            print(f"{self.name} cannot transfer data to {other_device.name} due to barriers or range.")

# Example usage
device_a = Device("Device A", (0, 0), 50)
device_b = Device("Device B", (30, 40), 50)

device_a.allowed_devices = ["Device B"]
device_b.allowed_devices = ["Device A"]

# Simulating different data transfer sizes
device_a.transfer_data(device_b, 100)  # Small data transfer
device_a.transfer_data(device_b, 600)  # Medium data transfer
device_a.transfer_data(device_b, 2000) # Large data transfer
