from dynamicmodulediscoveryandloading import discover_and_load_modules


def integrate_and_execute(discovered_modules, data):
    results = {}
    for name, module in discovered_modules.items():
        module.initialize()  # Initialize the module
        module.process_data(data)  # Process data using the module
        results[name] = module.get_results()  # Collect results
    return results

MODULES_DIRECTORY = 'C:/Users/HeadAdminKiriguya/Documents/AuraProject1'

def main():
    # Discover foreign modules
    foreign_modules = discover_and_load_modules(MODULES_DIRECTORY)
    
    # Data to process
    data = {'sample_data': 123}
    
    # Integrate and execute discovered modules
    results = integrate_and_execute(foreign_modules, data)
    
    print("Integration Results:", results)

if __name__ == "__main__":
    main()
