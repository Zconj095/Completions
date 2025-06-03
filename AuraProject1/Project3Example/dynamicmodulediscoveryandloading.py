import os
import importlib.util
from required_functions import *
def discover_and_load_modules(modules_dir):
    discovered_modules = {}
    for filename in os.listdir(modules_dir):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            module_path = os.path.join(modules_dir, filename)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Check if the module meets the integration criteria
            if all(hasattr(module, func) for func in REQUIRED_FUNCTIONS):
                discovered_modules[module_name] = module
    return discovered_modules
