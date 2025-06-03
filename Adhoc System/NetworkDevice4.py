class NetworkDevice:
    # ... existing NetworkDevice class ...

def display_menu():
    print("\nNetwork Device Interface")
    print("1. Connect Device")
    print("2. Transfer Data")
    print("3. Display Device Status")
    print("4. Exit")

def main():
    # Example setup
    devices = {
        "Device A": NetworkDevice("Device A", (0, 0), 50, 'WiFi', 500, 80, 20),
        "Device B": NetworkDevice("Device B", (30, 40), 50, 'WiFi', 500, 80, 20)
    }

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            # Connect devices (simplified example)
            device_name = input("Enter device name to connect: ")
            target_device = input("Enter target device name: ")
            if device_name in devices and target_device in devices:
                devices[device_name].allowed_devices.append(target_device)
                print(f"{device_name} is now allowed to connect with {target_device}")
            else:
                print("Device not found.")
        
        elif choice == "2":
            # Transfer data
            device_name = input("Enter source device name: ")
            target_device = input("Enter target device name: ")
            data_size = int(input("Enter data size (KB): "))
            if device_name in devices and target_device in devices:
                devices[device_name].transfer_data(devices[target_device], data_size)
            else:
                print("Device not found.")
        
        elif choice == "3":
            # Display device status
            device_name = input("Enter device name: ")
            if device_name in devices:
                device = devices[device_name]
                print(f"Device {device_name}: Position {device.position}, Range {device.range}")
            else:
                print("Device not found.")
        
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
