import time
import random
import math

class NetworkSimulation:
    def __init__(self, speed, wavelength, protocol="TCP"):
        self.speed = speed  # Speed of transmission in Kbps
        self.wavelength = wavelength  # Wavelength in nanometers (nm)
        self.protocol = protocol  # Network protocol (TCP/UDP)
        self.time_adjustment_factor = 1
        self.congestion_level = 0.1  # Network congestion (0-1)
        self.error_correction = True
        self.retransmission_attempts = 3
        
    def set_time_adjustment(self, day_factor):
        self.time_adjustment_factor = day_factor
        
    def set_network_conditions(self, congestion=0.1, error_correction=True):
        self.congestion_level = max(0, min(1, congestion))
        self.error_correction = error_correction
        
    def calculate_packet_loss_rate(self):
        """Calculate dynamic packet loss based on network conditions"""
        base_loss_rate = 0.001  # 0.1% base packet loss
        
        # Factor in wavelength (higher wavelength = more interference)
        wavelength_factor = self.wavelength / 1000
        
        # Factor in speed (higher speed = more potential for errors)
        speed_factor = math.log(self.speed + 1) / 10
        
        # Congestion impact
        congestion_factor = self.congestion_level * 0.05
        
        total_loss_rate = (base_loss_rate + wavelength_factor * 0.001 + 
                          speed_factor * 0.001 + congestion_factor) * self.time_adjustment_factor
        
        return min(total_loss_rate, 0.5)  # Cap at 50% loss
        
    def simulate_latency(self):
        """Simulate network latency in milliseconds"""
        base_latency = 10 + (self.wavelength / 100)  # Base latency
        congestion_latency = self.congestion_level * 50
        random_jitter = random.uniform(-5, 15)
        
        return max(1, base_latency + congestion_latency + random_jitter)
        
    def transfer_data(self, data_size):
        """Enhanced data transfer simulation with detailed metrics"""
        packet_size = 1.5  # More realistic packet size in KB (1500 bytes)
        total_packets = math.ceil(data_size / packet_size)
        packet_loss_rate = self.calculate_packet_loss_rate()
        
        # Simulate packet transmission
        successful_packets = 0
        retransmitted_packets = 0
        lost_packets = 0
        
        for packet in range(int(total_packets)):
            # Simulate packet loss
            if random.random() < packet_loss_rate:
                if self.protocol == "TCP" and self.error_correction:
                    # TCP retransmission
                    for attempt in range(self.retransmission_attempts):
                        if random.random() > packet_loss_rate:
                            successful_packets += 1
                            retransmitted_packets += 1
                            break
                    else:
                        lost_packets += 1
                else:
                    lost_packets += 1
            else:
                successful_packets += 1
        
        # Calculate transfer time
        base_transfer_time = (data_size * 8) / self.speed  # Convert KB to Kb
        latency = self.simulate_latency() / 1000  # Convert ms to seconds
        retransmission_overhead = retransmitted_packets * 0.1  # Additional time for retransmissions
        
        total_time = base_transfer_time + latency + retransmission_overhead
        
        # Calculate throughput
        successful_data = (successful_packets * packet_size)
        throughput = (successful_data * 8) / total_time if total_time > 0 else 0
        
        # Display results
        print(f"\n=== Network Transfer Report ===")
        print(f"Protocol: {self.protocol}")
        print(f"Data Size: {data_size} KB")
        print(f"Total Packets: {total_packets}")
        print(f"Successful: {successful_packets} ({successful_packets/total_packets*100:.1f}%)")
        print(f"Retransmitted: {retransmitted_packets}")
        print(f"Lost: {lost_packets} ({lost_packets/total_packets*100:.1f}%)")
        print(f"Transfer Time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} Kbps")
        print(f"Efficiency: {(throughput/self.speed)*100:.1f}%")
        print(f"Average Latency: {self.simulate_latency():.1f} ms")
        
        return {
            'successful_packets': successful_packets,
            'lost_packets': lost_packets,
            'total_time': total_time,
            'throughput': throughput
        }

# Example usage with enhanced features
if __name__ == "__main__":
    # Create network simulation
    network = NetworkSimulation(speed=100, wavelength=850, protocol="TCP")
    
    # Set various network conditions
    network.set_time_adjustment(day_factor=1.0)
    network.set_network_conditions(congestion=0.2, error_correction=True)
    
    # Test different scenarios
    print("Testing under normal conditions:")
    network.transfer_data(1000)
    
    print("\nTesting under high congestion:")
    network.set_network_conditions(congestion=0.7)
    network.transfer_data(1000)
    
    print("\nTesting UDP without error correction:")
    network.protocol = "UDP"
    network.set_network_conditions(congestion=0.2, error_correction=False)
    network.transfer_data(1000)
