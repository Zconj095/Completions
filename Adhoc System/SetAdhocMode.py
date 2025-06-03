import subprocess
import logging
import sys
import os
import re
import time
import json
import argparse
import threading
import signal
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import ipaddress
import socket
from io import StringIO
import numpy as np

# --- LOGGING SETUP: Only log to console during execution ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONSOLE OUTPUT MANAGEMENT ---
# Store original stdout/stderr for potential restoration
original_stdout = sys.stdout
original_stderr = sys.stderr

class NetworkMode(Enum):
    MANAGED = "managed"
    ADHOC = "ad-hoc"
    INFRASTRUCTURE = "infrastructure"

class SecurityType(Enum):
    OPEN = "open"
    WEP = "wep"
    WPA = "wpa"
    WPA2 = "wpa2"

class PowerMode(Enum):
    ON = "on"
    OFF = "off"
    AUTO = "auto"

@dataclass
class WirelessCapabilities:
    """Data class for wireless interface capabilities."""
    supported_modes: List[str] = field(default_factory=list)
    supported_channels: Dict[str, List[int]] = field(default_factory=dict)
    max_tx_power: Optional[str] = None
    encryption_support: List[str] = field(default_factory=list)
    frequency_ranges: List[str] = field(default_factory=list)

@dataclass
class NetworkConfig:
    """Enhanced data class for storing network configuration."""
    ip_address: str
    essid: Optional[str] = None
    channel: Optional[int] = None
    mode: NetworkMode = NetworkMode.ADHOC
    encryption: Optional[SecurityType] = None
    power: Optional[PowerMode] = None
    key: Optional[str] = None  # WEP/WPA key
    retry_count: int = 3
    timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.mode, str):
            self.mode = NetworkMode(self.mode)
        if isinstance(self.encryption, str) and self.encryption:
            self.encryption = SecurityType(self.encryption)
        if isinstance(self.power, str) and self.power:
            self.power = PowerMode(self.power)

class WindowsAdhocNetworkManager:
    def __init__(self, interface: str, config_file: Optional[str] = None):
        self.interface = interface
        self.original_config = None
        self.config_file = config_file or f"adhoc_backup_{interface.replace(' ', '_')}.json"
        self.capabilities: Optional[WirelessCapabilities] = None
        self.lock = threading.Lock()
        
        # Windows-specific channel support
        self.supported_channels = {
            '2.4GHz': list(range(1, 15)),  # Channels 1-14
            '5GHz': [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 
                    116, 120, 124, 128, 132, 136, 140, 149, 153, 157, 161, 165]
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    @contextmanager
    def _operation_lock(self):
        """Context manager for thread-safe operations."""
        self.lock.acquire()
        try:
            yield
        finally:
            self.lock.release()
    
    def _run_command(self, command: List[str], description: str, timeout: int = 30, 
                    retry_count: int = 1, shell: bool = True) -> Tuple[bool, str]:
        """Execute a Windows system command with enhanced error handling."""
        for attempt in range(retry_count):
            try:
                logger.info(f"Executing: {description} (attempt {attempt + 1}/{retry_count})")
                logger.debug(f"Command: {' '.join(command) if isinstance(command, list) else command}")
                
                if shell and isinstance(command, list):
                    command = ' '.join(command)
                
                result = subprocess.run(
                    command, 
                    shell=shell,
                    check=False,  # Don't raise on non-zero exit
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                
                logger.debug(f"Command output: {result.stdout}")
                if result.stderr:
                    logger.debug(f"Command stderr: {result.stderr}")
                
                return result.returncode == 0, result.stdout + result.stderr
                
            except subprocess.TimeoutExpired:
                logger.error(f"Command timed out while {description}")
                if attempt == retry_count - 1:
                    return False, "Timeout"
                time.sleep(2)
            except Exception as e:
                logger.error(f"Failed to {description} (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return False, str(e)
                time.sleep(2)
        
        return False, "All retry attempts failed"
    
    def _check_dependencies(self) -> bool:
        """Check if required Windows tools are available."""
        required_tools = ['netsh', 'ipconfig']
        missing_tools = []
        
        for tool in required_tools:
            if not self._command_exists(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False
        
        return True
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in Windows."""
        try:
            result = subprocess.run(['where', command], capture_output=True, timeout=5, shell=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_wireless_profiles(self) -> List[str]:
        """Get list of wireless profiles on Windows."""
        success, output = self._run_command(['netsh', 'wlan', 'show', 'profiles'], 
                                          "get wireless profiles")
        profiles = []
        if success:
            for line in output.split('\n'):
                if 'All User Profile' in line:
                    # Extract profile name
                    match = re.search(r': (.+)$', line.strip())
                    if match:
                        profiles.append(match.group(1).strip())
        return profiles
    
    def _get_wireless_interfaces(self) -> List[str]:
        """Get list of wireless interfaces on Windows."""
        success, output = self._run_command(['netsh', 'wlan', 'show', 'interfaces'], 
                                          "get wireless interfaces")
        interfaces = []
        if success:
            current_interface = None
            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('Name'):
                    match = re.search(r': (.+)$', line)
                    if match:
                        current_interface = match.group(1).strip()
                        if current_interface:
                            interfaces.append(current_interface)
        return interfaces
    
    def _validate_ip_address(self, ip_address: str) -> bool:
        """Enhanced IP address validation using ipaddress module."""
        try:
            network = ipaddress.ip_network(ip_address, strict=False)
            
            if network.is_private:
                logger.info(f"Using private IP range: {network}")
            elif network.is_loopback:
                logger.warning(f"Using loopback address: {network}")
                return False
            elif network.is_multicast:
                logger.error(f"Multicast addresses not allowed: {network}")
                return False
            
            if network.prefixlen < 8:
                logger.warning(f"Very large subnet: /{network.prefixlen}")
            elif network.prefixlen > 30:
                logger.warning(f"Very small subnet: /{network.prefixlen}")
            
            return True
            
        except ValueError as e:
            logger.error(f"Invalid IP address format: {ip_address}. Error: {e}")
            return False
    
    def _check_admin_privileges(self) -> bool:
        """Check if running with administrator privileges on Windows."""
        try:
            # Try to access a system directory that requires admin rights
            test_path = r"C:\Windows\System32\drivers\etc"
            test_file = os.path.join(test_path, "hosts")
            with open(test_file, 'r') as f:
                pass
            return True
        except (PermissionError, OSError):
            logger.error("Administrator privileges required. Please run as Administrator.")
            logger.info("Right-click on Command Prompt/PowerShell and select 'Run as administrator'")
            return False
    
    def _backup_current_config(self) -> bool:
        """Backup current Windows network configuration."""
        try:
            backup_data = {
                'timestamp': time.time(),
                'interface': self.interface,
                'version': '2.0_windows'
            }
            
            # Get current interface configuration
            success, output = self._run_command(['netsh', 'interface', 'ip', 'show', 'config', f'name="{self.interface}"'], 
                                              "get interface IP configuration")
            if success:
                backup_data['ip_config'] = output
            
            # Get wireless configuration
            success, output = self._run_command(['netsh', 'wlan', 'show', 'interfaces'], 
                                              "get wireless configuration")
            if success:
                backup_data['wireless_config'] = output
            
            # Save backup
            backup_path = Path(self.config_file)
            backup_path.parent.mkdir(exist_ok=True)
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.original_config = backup_data
            logger.info(f"Configuration backed up to {self.config_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not backup configuration: {e}")
            return False
    
    def _create_adhoc_profile(self, config: NetworkConfig) -> bool:
        """Create an ad-hoc network profile on Windows."""
        profile_name = f"adhoc_{config.essid or 'network'}"
        
        # Create XML profile for ad-hoc network
        xml_content = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{profile_name}</name>
    <SSIDConfig>
        <SSID>
            <name>{config.essid or 'adhoc_network'}</name>
        </SSID>
        <connectionType>IBSS</connectionType>
    </SSIDConfig>
    <connectionMode>manual</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>{"open" if not config.encryption else config.encryption.value}</authentication>
                <encryption>{"none" if not config.encryption else "WEP" if config.encryption == SecurityType.WEP else "TKIP"}</encryption>
                <useOneX>false</useOneX>
            </authEncryption>"""
        
        if config.encryption and config.key:
            xml_content += f"""
            <sharedKey>
                <keyType>networkKey</keyType>
                <protected>false</protected>
                <keyMaterial>{config.key}</keyMaterial>
            </sharedKey>"""
        
        xml_content += """
        </security>
    </MSM>
</WLANProfile>"""
        
        # Save profile to temporary file
        profile_file = f"temp_adhoc_profile_{int(time.time())}.xml"
        try:
            with open(profile_file, 'w') as f:
                f.write(xml_content)
            
            # Add profile using netsh
            success, output = self._run_command(
                ['netsh', 'wlan', 'add', 'profile', f'filename="{profile_file}"', f'interface="{self.interface}"'],
                f"add ad-hoc profile {profile_name}"
            )
            
            # Clean up temporary file
            try:
                os.remove(profile_file)
            except Exception:
                pass
            
            if success:
                logger.info(f"Created ad-hoc profile: {profile_name}")
                return True
            else:
                logger.error(f"Failed to create ad-hoc profile: {output}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating ad-hoc profile: {e}")
            try:
                os.remove(profile_file)
            except Exception:
                pass
            return False
    
    def set_adhoc_mode(self, config: NetworkConfig) -> bool:
        """Set up ad-hoc mode on Windows."""
        with self._operation_lock():
            # Pre-flight checks
            if not self._comprehensive_validation(config):
                return False
            
            try:
                # Backup current configuration
                if not self._backup_current_config():
                    logger.warning("Could not backup current configuration")
                
                # Execute configuration steps
                if self._execute_windows_configuration(config):
                    logger.info(f"Successfully configured {self.interface} in ad-hoc mode")
                    self._display_windows_status()
                    return True
                else:
                    logger.error("Configuration failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Unexpected error during configuration: {e}")
                return False
    
    def _comprehensive_validation(self, config: NetworkConfig) -> bool:
        """Comprehensive pre-configuration validation for Windows."""
        validators = [
            (self._check_dependencies, "checking dependencies"),
            (lambda: self._validate_ip_address(config.ip_address), "validating IP address"),
            (lambda: self._validate_channel(config.channel) if config.channel else True, "validating channel"),
            (lambda: self._validate_essid(config.essid) if config.essid else True, "validating ESSID"),
            (self._check_admin_privileges, "checking privileges"),
            (self._check_interface_exists, "checking interface existence")
        ]
        
        for validator, description in validators:
            try:
                if not validator():
                    logger.error(f"Validation failed: {description}")
                    return False
            except Exception as e:
                logger.error(f"Validation error ({description}): {e}")
                return False
        
        return True
    
    def _execute_windows_configuration(self, config: NetworkConfig) -> bool:
        """Execute Windows-specific ad-hoc configuration."""
        steps = []
        
        # Step 1: Disconnect from any current network
        steps.append((
            ['netsh', 'wlan', 'disconnect', f'interface="{self.interface}"'],
            f"disconnect {self.interface} from current network"
        ))
        
        # Step 2: Create and connect to ad-hoc profile
        if config.essid:
            if self._create_adhoc_profile(config):
                profile_name = f"adhoc_{config.essid}"
                steps.append((
                    ['netsh', 'wlan', 'connect', f'name="{profile_name}"', f'interface="{self.interface}"'],
                    f"connect to ad-hoc network {config.essid}"
                ))
        
        # Step 3: Set IP address
        ip_parts = config.ip_address.split('/')
        ip_addr = ip_parts[0]
        
        # Calculate subnet mask from CIDR
        if len(ip_parts) > 1:
            prefix_len = int(ip_parts[1])
            subnet_mask = str(ipaddress.IPv4Network(f"0.0.0.0/{prefix_len}", strict=False).netmask)
        else:
            subnet_mask = "255.255.255.0"
        
        steps.append((
            ['netsh', 'interface', 'ip', 'set', 'address', f'name="{self.interface}"', 
             'static', ip_addr, subnet_mask],
            f"set IP address {config.ip_address} on {self.interface}"
        ))
        
        # Execute all steps
        for command, description in steps:
            success, output = self._run_command(command, description, timeout=config.timeout)
            if not success:
                logger.warning(f"Step failed: {description}")
                logger.debug(f"Output: {output}")
            time.sleep(1)  # Small delay between steps
        
        return self._verify_windows_configuration(config)
    
    def _verify_windows_configuration(self, config: NetworkConfig) -> bool:
        """Verify Windows ad-hoc configuration."""
        verifications = []
        
        # Check IP configuration
        success, output = self._run_command(['ipconfig', '/all'], "verify IP configuration")
        if success:
            ip_configured = config.ip_address.split('/')[0] in output
            verifications.append(("IP address", ip_configured))
        
        # Check wireless connection
        success, output = self._run_command(['netsh', 'wlan', 'show', 'interfaces'], 
                                          "verify wireless configuration")
        if success:
            if config.essid:
                essid_connected = config.essid in output
                verifications.append(("ESSID connection", essid_connected))
            
            # Check if connected to ad-hoc (IBSS)
            adhoc_mode = 'IBSS' in output or 'ad hoc' in output.lower()
            verifications.append(("Ad-hoc mode", adhoc_mode))
        
        # Report verification results
        all_passed = True
        for check, passed in verifications:
            status = "✓" if passed else "✗"
            logger.info(f"Verification {status} {check}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def _display_windows_status(self):
        """Display comprehensive Windows interface status."""
        try:
            print("\n" + "="*80)
            print("🌐 WINDOWS INTERFACE STATUS")
            print("="*80)
            
            # Interface information
            success, output = self._run_command(['ipconfig', '/all'], "get interface status")
            if success:
                print("\n📡 Interface Configuration:")
                print("-" * 40)
                # Filter output for relevant interface
                lines = output.split('\n')
                in_relevant_section = False
                for line in lines:
                    if self.interface in line or in_relevant_section:
                        print(line)
                        if line.strip() == "":
                            in_relevant_section = False
                        else:
                            in_relevant_section = True
            
            # Wireless information
            success, output = self._run_command(['netsh', 'wlan', 'show', 'interfaces'], 
                                              "get wireless status")
            if success:
                print("\n📶 Wireless Status:")
                print("-" * 40)
                print(output)
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Could not display status: {e}")
    
    def _validate_channel(self, channel: int) -> bool:
        """Validate wireless channel for Windows."""
        # Basic validation - Windows will handle specific capability checking
        if channel < 1 or channel > 165:
            logger.error(f"Invalid channel {channel}. Must be between 1-165")
            return False
        return True
    
    def _validate_essid(self, essid: str) -> bool:
        """Validate ESSID for Windows."""
        if not essid or len(essid) > 32:
            logger.error(f"Invalid ESSID: '{essid}'. Must be 1-32 characters")
            return False
        
        if any(ord(c) < 32 and c not in '\t\n\r' for c in essid):
            logger.error(f"ESSID contains invalid control characters: '{essid}'")
            return False
        
        return True
    
    def _check_interface_exists(self) -> bool:
        """Check if wireless interface exists on Windows."""
        interfaces = self._get_wireless_interfaces()
        if self.interface in interfaces:
            logger.info(f"Interface {self.interface} found")
            return True
        else:
            logger.error(f"Interface {self.interface} not found")
            logger.info(f"Available interfaces: {interfaces}")
            return False
    
    def restore_original_config(self) -> bool:
        """Restore original Windows configuration."""
        if not os.path.exists(self.config_file):
            logger.error(f"No backup file found: {self.config_file}")
            return False
        
        try:
            # Reset to DHCP
            success, _ = self._run_command(
                ['netsh', 'interface', 'ip', 'set', 'address', f'name="{self.interface}"', 'dhcp'],
                f"reset {self.interface} to DHCP"
            )
            
            if success:
                logger.info("Network interface reset to DHCP")
            
            # Clean up backup file
            try:
                os.remove(self.config_file)
                logger.info("Backup file cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove backup file: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error restoring configuration: {e}")
            return False
    
    def get_interface_info(self) -> Dict:
        """Get comprehensive Windows interface information."""
        info = {
            'interface': self.interface,
            'exists': False,
            'current_ip': None,
            'wireless_status': None,
            'available_interfaces': []
        }
        
        try:
            # Get available interfaces
            info['available_interfaces'] = self._get_wireless_interfaces()
            info['exists'] = self.interface in info['available_interfaces']
            
            if info['exists']:
                # Get current IP
                success, output = self._run_command(['ipconfig'], "get current IP config")
                if success:
                    # Parse IP for this interface
                    ip_match = re.search(r'IPv4 Address[.\s]*:\s*([0-9.]+)', output)
                    if ip_match:
                        info['current_ip'] = ip_match.group(1)
                
                # Get wireless status
                success, output = self._run_command(['netsh', 'wlan', 'show', 'interfaces'], 
                                                  "get wireless status")
                if success:
                    info['wireless_status'] = output
        
        except Exception as e:
            logger.error(f"Error getting interface info: {e}")
        
        return info

def create_windows_parser() -> argparse.ArgumentParser:
    """Create Windows-specific command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🌐 Windows Ad-hoc Network Manager v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic ad-hoc setup:
    python SetAdhocMode.py "Wi-Fi" 192.168.1.1/24 --essid MyAdhocNet --channel 6
  
  With WEP encryption:
    python SetAdhocMode.py "Wi-Fi" 10.0.0.1/8 --essid TestNet --encryption wep --key 1234567890
  
  Management operations:
    python SetAdhocMode.py "Wi-Fi" --restore
    python SetAdhocMode.py "Wi-Fi" --status
    python SetAdhocMode.py "Wi-Fi" --info
    python SetAdhocMode.py "Wi-Fi" --list-interfaces
        """
    )
    
    parser.add_argument('interface', help='Network interface name (e.g., "Wi-Fi")')
    parser.add_argument('ip_address', nargs='?', help='IP address with CIDR notation (e.g., 192.168.1.1/24)')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--essid', '-e', help='Network name for the ad-hoc network')
    config_group.add_argument('--channel', '-c', type=int, help='Wireless channel')
    config_group.add_argument('--encryption', choices=['open', 'wep'], help='Encryption type (Windows supports open/WEP for ad-hoc)')
    config_group.add_argument('--key', help='Network key (for WEP encryption)')
    
    # Operation options
    ops_group = parser.add_argument_group('Operation Options')
    ops_group.add_argument('--restore', '-r', action='store_true', help='Restore original configuration')
    ops_group.add_argument('--status', '-s', action='store_true', help='Show current interface status')
    ops_group.add_argument('--info', '-i', action='store_true', help='Show comprehensive interface information')
    ops_group.add_argument('--list-interfaces', '-l', action='store_true', help='List available wireless interfaces')
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--config-file', help='Custom backup configuration file path')
    advanced_group.add_argument('--timeout', type=int, default=30, help='Command timeout in seconds')
    advanced_group.add_argument('--retry-count', type=int, default=3, help='Number of retries for failed commands')
    advanced_group.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    advanced_group.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser

def main():
    """Enhanced main function for Windows."""
    parser = create_windows_parser()
    
    # Handle case where no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # If argparse exits due to --help or invalid args, handle gracefully
        if e.code == 0:  # --help was called
            return 0
        else:  # Invalid arguments
            logger.error("Invalid command line arguments. Use --help for usage information.")
            return e.code

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Create manager instance
    manager = WindowsAdhocNetworkManager(args.interface, config_file=args.config_file)

    try:
        # Handle information operations
        if args.info:
            info = manager.get_interface_info()
            print(json.dumps(info, indent=2, default=str))
            return 0

        if args.list_interfaces:
            interfaces = manager._get_wireless_interfaces()
            print("\n📡 Available Wireless Interfaces:")
            print("-" * 40)
            for interface in interfaces:
                print(f"  • {interface}")
            return 0

        if args.restore:
            success = manager.restore_original_config()
            return 0 if success else 1

        if args.status:
            manager._display_windows_status()
            return 0

        # Validate required arguments for ad-hoc setup
        if not args.ip_address:
            print("Error: ip_address is required for ad-hoc mode setup")
            print("Use --help for usage information")
            return 1

        try:
            config = NetworkConfig(
                ip_address=args.ip_address,
                essid=args.essid,
                channel=args.channel,
                encryption=SecurityType(args.encryption) if args.encryption else None,
                key=args.key,
                retry_count=args.retry_count,
                timeout=args.timeout
            )
        except ValueError as e:
            logger.error(f"Invalid configuration parameter: {e}")
            return 1

        success = manager.set_adhoc_mode(config)
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 1
    except SystemExit as e:
        return getattr(e, 'code', 1)
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        return 1
    finally:
        # Ensure stdout/stderr are restored
        sys.stdout = original_stdout
        sys.stderr = original_stderr

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except SystemExit:
        # Let SystemExit propagate naturally
        raise
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        # --- MULTI-DIMENSIONAL HADAMARD MATRICES UTILITIES ---


        def hadamard_matrix(n: int) -> np.ndarray:
            """Generate a Hadamard matrix of order n (n must be 1, 2, 4, 8, ...)."""
            if n == 1:
                return np.array([[1]])
            elif n % 2 != 0:
                raise ValueError("Hadamard matrix order must be a power of 2")
            else:
                H = hadamard_matrix(n // 2)
                return np.block([[H, H], [H, -H]])

        def intertwine_hadamard_across_tuples(tuple_lists: List[List[Tuple[int, ...]]], subset_arrays: List[np.ndarray]) -> List[np.ndarray]:
            """
            For each tuple list and subset array, generate a multi-dimensional Hadamard matrix,
            intertwine them by tensor (Kronecker) product, and nest the results.
            """
            results = []
            for tuples, subset in zip(tuple_lists, subset_arrays):
                # For each tuple, create a Hadamard matrix of size matching the tuple length or subset shape
                nested = []
                for t in tuples:
                    size = 2 ** len(t)
                    H = hadamard_matrix(size)
                    # Intertwine with subset array using Kronecker product
                    entwined = np.kron(H, subset)
                    nested.append(entwined)
                results.append(nested)
            return results

        def multi_complex_hadamard_interconnection(arrays: List[np.ndarray]) -> np.ndarray:
            """
            Recursively intertwine a list of arrays using Kronecker product to form a multi-complex Hadamard structure.
            """
            if len(arrays) == 1:
                return arrays[0]
            else:
                return np.kron(arrays[0], multi_complex_hadamard_interconnection(arrays[1:]))

        # Example usage (for demonstration, can be removed in production):
        if __name__ == "__main__":
            # Example: intertwine Hadamard matrices across tuple lists and subset arrays
            tuple_lists = [
                [(0, 1), (1, 0)],
                [(1, 1)]
            ]
            subset_arrays = [
                np.array([[1, 0], [0, 1]]),
                np.array([[0, 1], [1, 0]])
            ]
            intertwined = intertwine_hadamard_across_tuples(tuple_lists, subset_arrays)
            # Multi-complex interconnection
            multi_complex = multi_complex_hadamard_interconnection([hadamard_matrix(2), hadamard_matrix(2), hadamard_matrix(2)])
            print("Intertwined Hadamard Sets:", intertwined)
            print("Multi-Complex Hadamard Structure:\n", multi_complex)
