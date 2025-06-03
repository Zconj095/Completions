class NetworkDevice:
    def __init__(self, name, connection_type, position=(0,0), range=50, protocol='WiFi', wavelength=500, signal_strength=80, noise_level=20):
        self.name = name
        self.connection_type = connection_type  # 'Ethernet' or 'Wireless'
        self.position = position
        self.range = range if connection_type == 'Wireless' else None
        self.protocol = protocol
        self.wavelength = wavelength
        self.signal_strength = signal_strength
        self.noise_level = noise_level
        self.allowed_devices = []
        self.packet_delay = 0.0012 if connection_type == 'Wireless' else 0.0001  # Smaller delay for Ethernet
        self.packet_loss_rate = self.calculate_packet_loss_rate()
        self.transfer_rate = self.calculate_transfer_rate()  # Mbps

    # ... existing methods ...

    def calculate_transfer_rate(self):
        if self.connection_type == 'Ethernet':
            return 1000  # Ethernet typically offers higher speeds, set to 1 Gbps for example
        else:
            # ... existing wireless transfer rate calculation ...

    def calculate_packet_loss_rate(self):
        if self.connection_type == 'Ethernet':
            return 0.001  # Lower packet loss rate for Ethernet
        else:
            # ... existing wireless packet loss rate calculation ...

# ... existing main function and user interface ...

if __name__ == "__main__":
    main()
