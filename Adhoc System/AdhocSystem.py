import random
import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

class ConnectionType(Enum):
    ETHERNET = "Ethernet"
    WIRELESS = "Wireless"
    BLUETOOTH = "Bluetooth"
    FIBER_OPTIC = "Fiber_Optic"

class DeviceStatus(Enum):
    ACTIVE = "Active"
    IDLE = "Idle"
    BUSY = "Busy"
    OFFLINE = "Offline"

@dataclass
class DataPacket:
    data: Any
    source: str
    destination: str
    timestamp: float
    priority: int = 1  # 1 = low, 5 = high
    size_bytes: int = 1024

class NetworkDevice:
    def __init__(self, name: str, connection_type: ConnectionType, position: tuple, 
                 range: float, is_edge_device: bool = False, max_connections: int = 10):
        self.name = name
        self.connection_type = connection_type
        self.position = position
        self.range = range
        self.is_edge_device = is_edge_device
        self.max_connections = max_connections
        self.connected_devices: Dict[str, 'NetworkDevice'] = {}
        self.sync_timer = 0
        self.status = DeviceStatus.IDLE
        self.bandwidth_mbps = self._get_default_bandwidth()
        self.data_queue: List[DataPacket] = []
        self.processing_power = 100 if is_edge_device else 50
        self.battery_level = 100.0
        self.lock = threading.Lock()

    def _get_default_bandwidth(self) -> float:
        bandwidth_map = {
            ConnectionType.ETHERNET: 1000.0,
            ConnectionType.FIBER_OPTIC: 10000.0,
            ConnectionType.WIRELESS: 300.0,
            ConnectionType.BLUETOOTH: 2.0
        }
        return bandwidth_map.get(self.connection_type, 100.0)

    def connect_device(self, other_device: 'NetworkDevice') -> bool:
        """Connect with another device if within range and capacity allows."""
        if len(self.connected_devices) >= self.max_connections:
            print(f"Connection failed: {self.name} has reached maximum connections.")
            return False
            
        if not self.is_in_range(other_device):
            print(f"Connection failed: {other_device.name} is out of range.")
            return False
            
        with self.lock:
            self.connected_devices[other_device.name] = other_device
            other_device.connected_devices[self.name] = self
            
        print(f"{self.name} is now connected to {other_device.name}.")
        return True

    def disconnect_device(self, other_device: 'NetworkDevice') -> bool:
        """Disconnect from another device."""
        with self.lock:
            if other_device.name in self.connected_devices:
                del self.connected_devices[other_device.name]
                del other_device.connected_devices[self.name]
                print(f"{self.name} disconnected from {other_device.name}.")
                return True
        return False

    def is_in_range(self, other_device: 'NetworkDevice') -> bool:
        """Check if another device is within connection range."""
        if self.connection_type in [ConnectionType.ETHERNET, ConnectionType.FIBER_OPTIC]:
            return True  # Wired connections are not limited by range
            
        distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                   (self.position[1] - other_device.position[1]) ** 2) ** 0.5
        return distance <= self.range

    def process_data(self, packet: DataPacket) -> Optional[DataPacket]:
        """Process data if the device is an edge computing device."""
        if not self.is_edge_device:
            print(f"{self.name} is not an edge computing device.")
            return None
            
        if self.battery_level < 10:
            print(f"{self.name} has low battery and cannot process data.")
            return None
            
        self.status = DeviceStatus.BUSY
        processing_time = packet.size_bytes / (self.processing_power * 1000)  # seconds
        time.sleep(processing_time)
        
        # Consume battery
        self.battery_level = max(0, self.battery_level - (processing_time * 2))
        
        processed_packet = DataPacket(
            data=f"Processed {packet.data} by {self.name}",
            source=self.name,
            destination=packet.destination,
            timestamp=time.time(),
            priority=packet.priority,
            size_bytes=packet.size_bytes
        )
        
        self.status = DeviceStatus.IDLE
        print(f"Data processed by {self.name}: {processed_packet.data}")
        return processed_packet

    def transfer_data(self, other_device: 'NetworkDevice', data: Any, 
                     priority: int = 1, size_bytes: int = 1024) -> Optional[DataPacket]:
        """Transfer data to another connected device."""
        if other_device.name not in self.connected_devices:
            print(f"Transfer failed: {other_device.name} is not connected.")
            return None
            
        packet = DataPacket(
            data=data,
            source=self.name,
            destination=other_device.name,
            timestamp=time.time(),
            priority=priority,
            size_bytes=size_bytes
        )
        
        transfer_time = self.calculate_transfer_time(packet, other_device)
        self.sync_transfer(other_device)
        
        print(f"Transferring data from {self.name} to {other_device.name}...")
        print(f"Estimated transfer time: {transfer_time:.2f}s")
        
        time.sleep(transfer_time)
        
        # Add to destination's queue
        other_device.data_queue.append(packet)
        
        return other_device.process_data(packet)

    def calculate_transfer_time(self, packet: DataPacket, other_device: 'NetworkDevice') -> float:
        """Calculate data transfer time based on bandwidth and packet size."""
        effective_bandwidth = min(self.bandwidth_mbps, other_device.bandwidth_mbps)
        signal_quality = self.calculate_signal_quality(other_device)
        adjusted_bandwidth = effective_bandwidth * (signal_quality / 100)
        
        # Convert to bytes per second and calculate time
        bandwidth_bps = adjusted_bandwidth * 1_000_000 / 8
        return packet.size_bytes / bandwidth_bps

    def sync_transfer(self, other_device: 'NetworkDevice') -> None:
        """Synchronize transfer to optimize timing based on delays and signals."""
        signal_quality = self.calculate_signal_quality(other_device)
        
        if self.connection_type in [ConnectionType.ETHERNET, ConnectionType.FIBER_OPTIC]:
            self.sync_timer = max(1, 100 / signal_quality)
            print(f"Synchronization timer set to {self.sync_timer:.1f}ms for {self.name}")
        else:
            self.sync_timer = max(5, 200 / signal_quality)
            print(f"Wireless sync timer set to {self.sync_timer:.1f}ms for {self.name}")

    def calculate_signal_quality(self, other_device: 'NetworkDevice') -> float:
        """Calculate signal quality based on connection type and distance."""
        if self.connection_type in [ConnectionType.ETHERNET, ConnectionType.FIBER_OPTIC]:
            return random.uniform(90, 100)
            
        distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                   (self.position[1] - other_device.position[1]) ** 2) ** 0.5
        
        # Signal degrades with distance
        max_quality = 100 - (distance / self.range) * 30
        return random.uniform(max(20, max_quality - 10), max_quality)

    def get_network_topology(self) -> Dict[str, List[str]]:
        """Get the current network topology from this device's perspective."""
        topology = {self.name: list(self.connected_devices.keys())}
        
        for device in self.connected_devices.values():
            topology[device.name] = list(device.connected_devices.keys())
            
        return topology

    def find_route_to_device(self, target_device_name: str, visited: set = None) -> Optional[List[str]]:
        """Find a route to a target device using BFS."""
        if visited is None:
            visited = set()
            
        if target_device_name in self.connected_devices:
            return [self.name, target_device_name]
            
        visited.add(self.name)
        
        for device_name, device in self.connected_devices.items():
            if device_name not in visited:
                route = device.find_route_to_device(target_device_name, visited.copy())
                if route:
                    return [self.name] + route
                    
        return None

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of the device."""
        return {
            "name": self.name,
            "status": self.status.value,
            "connection_type": self.connection_type.value,
            "position": self.position,
            "range": self.range,
            "is_edge_device": self.is_edge_device,
            "connected_devices": len(self.connected_devices),
            "max_connections": self.max_connections,
            "bandwidth_mbps": self.bandwidth_mbps,
            "battery_level": f"{self.battery_level:.1f}%",
            "queue_size": len(self.data_queue),
            "processing_power": self.processing_power
        }

# Example usage with enhanced features
if __name__ == "__main__":
    # Create devices with different connection types
    device_a = NetworkDevice("EdgeServer", ConnectionType.ETHERNET, (0, 0), 50, True)
    device_b = NetworkDevice("Laptop", ConnectionType.WIRELESS, (30, 40), 50, False)
    device_c = NetworkDevice("IoTSensor", ConnectionType.BLUETOOTH, (10, 10), 20, False)
    
    # Connect devices
    device_a.connect_device(device_b)
    device_a.connect_device(device_c)
    device_b.connect_device(device_c)
    
    # Transfer data with different priorities
    device_a.transfer_data(device_b, "Critical sensor data", priority=5, size_bytes=2048)
    device_c.transfer_data(device_a, "Temperature reading: 23°C", priority=2, size_bytes=512)
    
    # Get network topology
    print("\nNetwork Topology:")
    topology = device_a.get_network_topology()
    for device, connections in topology.items():
        print(f"{device}: {connections}")
    
    # Find route
    route = device_c.find_route_to_device("Laptop")
    print(f"\nRoute from {device_c.name} to Laptop: {route}")
    
    # Status reports
    print(f"\nDevice A Status: {device_a.get_status_report()}")
