import json
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import neurokit2 as nk
    import numpy as np
except ImportError:
    logger.warning("Neurokit2 not available. Install with: pip install neurokit2")
    nk = None

from pydantic import BaseModel, Field, field_validator

class CustomBluetoothAdapter:
    """Custom Bluetooth adapter implementation"""
    
    def __init__(self):
        self.available = True
        self.connected_devices = {}
    
    def scan_devices(self, duration: int = 5) -> List[Dict[str, str]]:
        """Simulate scanning for Bluetooth devices"""
        logger.info(f"Scanning for devices for {duration} seconds...")
        time.sleep(1)  # Simulate scan time
        
        # Return mock devices
        return [
            {"mac": "00:11:22:33:44:55", "name": "HeartRate_Monitor_1"},
            {"mac": "AA:BB:CC:DD:EE:FF", "name": "HeartRate_Monitor_2"},
            {"mac": "12:34:56:78:90:AB", "name": "Fitness_Tracker"}
        ]
    
    def connect(self, mac_address: str) -> bool:
        """Connect to a Bluetooth device"""
        try:
            logger.info(f"Connecting to device: {mac_address}")
            time.sleep(0.5)  # Simulate connection time
            
            # Simulate connection success/failure
            if mac_address in ["00:11:22:33:44:55", "AA:BB:CC:DD:EE:FF"]:
                self.connected_devices[mac_address] = {
                    "connected_at": time.time(),
                    "status": "connected"
                }
                logger.info(f"Successfully connected to {mac_address}")
                return True
            else:
                logger.error(f"Failed to connect to {mac_address}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self, mac_address: str) -> bool:
        """Disconnect from a Bluetooth device"""
        try:
            if mac_address in self.connected_devices:
                del self.connected_devices[mac_address]
                logger.info(f"Disconnected from {mac_address}")
                return True
            return False
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
            return False
    
    def read_data(self, mac_address: str) -> Optional[bytes]:
        """Read data from connected device"""
        if mac_address not in self.connected_devices:
            return None
            
        # Simulate heart rate data (replace with actual implementation)
        heart_rate = random.randint(60, 100)
        return f"HR:{heart_rate}".encode('utf-8')
    
    def is_connected(self, mac_address: str) -> bool:
        """Check if device is connected"""
        return mac_address in self.connected_devices

class User(BaseModel):
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(male|female|other)$")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        return v.strip().title()

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "John Doe",
                "age": 35,
                "gender": "male"
            }
        }
    }

class HeartRateData(BaseModel):
    heart_rate: float = Field(..., ge=30, le=220)
    timestamp: float
    user_id: Optional[str] = None

class HRVMetrics(BaseModel):
    rmssd: Optional[float] = None
    pnn50: Optional[float] = None
    lf_hf_ratio: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)

class HeartRateSensor:
    def __init__(self, sensor_mac: Optional[str] = None):
        self.sensor_mac = sensor_mac
        self.bluetooth = CustomBluetoothAdapter()
        self.is_connected = False
        self._connect()

    def _connect(self):
        """Connect to Bluetooth heart rate sensor"""
        try:
            if self.sensor_mac:
                logger.info(f"Attempting to connect to sensor: {self.sensor_mac}")
                self.is_connected = self.bluetooth.connect(self.sensor_mac)
                if self.is_connected:
                    logger.info("Successfully connected to heart rate sensor")
                else:
                    logger.error("Failed to connect to heart rate sensor")
            else:
                logger.warning("No sensor MAC address provided")
        except Exception as e:
            logger.error(f"Failed to connect to sensor: {e}")
            self.is_connected = False

    def get_data(self) -> Optional[HeartRateData]:
        """Get heart rate data from sensor"""
        if not self.is_connected:
            logger.warning("Sensor not connected")
            return None
            
        try:
            # Read data from custom Bluetooth adapter
            raw_data = self.bluetooth.read_data(self.sensor_mac)
            if raw_data:
                # Parse heart rate from data (format: "HR:75")
                data_str = raw_data.decode('utf-8')
                if data_str.startswith("HR:"):
                    bpm = float(data_str.split(":")[1])
                    timestamp = time.time()
                    
                    return HeartRateData(
                        heart_rate=bpm,
                        timestamp=timestamp
                    )
            return None
        except Exception as e:
            logger.error(f"Error reading sensor data: {e}")
            return None

    def scan_devices(self) -> List[Dict[str, str]]:
        """Scan for available Bluetooth devices"""
        return self.bluetooth.scan_devices()

    def save_data(self, data: Dict[str, Any], file_path: str):
        """Save heart rate data to file"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=2, default=str)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def disconnect(self):
        """Disconnect from sensor"""
        if self.sensor_mac and self.is_connected:
            try:
                self.bluetooth.disconnect(self.sensor_mac)
                self.is_connected = False
                logger.info("Disconnected from sensor")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

class HRVAnalyzer:
    """Heart Rate Variability analysis using NeuroKit2"""
    
    @staticmethod
    def extract_hrv(hr_time_series: List[float], sampling_rate: int = 1000) -> Optional[HRVMetrics]:
        """Extract HRV features from heart rate time series"""
        if not nk:
            logger.error("NeuroKit2 not available for HRV analysis")
            return None
            
        try:
            if len(hr_time_series) < 10:
                logger.warning("Insufficient data for HRV analysis")
                return None
                
            # Convert to numpy array
            hr_array = np.array(hr_time_series)
            
            # Use NeuroKit2 to analyze HRV
            hrv_indices = nk.hrv_time(hr_array, sampling_rate=sampling_rate)
            hrv_freq = nk.hrv_frequency(hr_array, sampling_rate=sampling_rate)
            
            return HRVMetrics(
                rmssd=hrv_indices.get("HRV_RMSSD", [None])[0],
                pnn50=hrv_indices.get("HRV_pNN50", [None])[0],
                lf_hf_ratio=hrv_freq.get("HRV_LF", [None])[0] / hrv_freq.get("HRV_HF", [None])[0] 
                            if hrv_freq.get("HRV_HF", [None])[0] else None
            )
            
        except Exception as e:
            logger.error(f"Error in HRV analysis: {e}")
            return None

class BiometricSession:
    """Manage a complete biometric monitoring session"""
    
    def __init__(self, user: User, sensor_mac: Optional[str] = None):
        self.user = user
        self.sensor = HeartRateSensor(sensor_mac)
        self.session_data: List[HeartRateData] = []
        self.session_start = datetime.now()
        
    def collect_data(self, duration: int = 60, interval: float = 1.0):
        """Collect heart rate data for specified duration"""
        logger.info(f"Starting data collection for {duration} seconds")
        
        end_time = time.time() + duration
        while time.time() < end_time:
            data = self.sensor.get_data()
            if data:
                data.user_id = self.user.id
                self.session_data.append(data)
                logger.info(f"HR: {data.heart_rate} BPM")
            time.sleep(interval)
            
        logger.info(f"Collected {len(self.session_data)} data points")
        
    def analyze_session(self) -> Optional[HRVMetrics]:
        """Analyze collected heart rate data for HRV metrics"""
        if not self.session_data:
            logger.warning("No data to analyze")
            return None
            
        hr_values = [data.heart_rate for data in self.session_data]
        return HRVAnalyzer.extract_hrv(hr_values)
        
    def save_session(self, file_path: Optional[str] = None):
        """Save session data and analysis"""
        if not file_path:
            timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
            file_path = f"biometric_session_{timestamp}.json"
            
        session_summary = {
            "user": self.user.model_dump(),
            "session_start": self.session_start.isoformat(),
            "data_points": len(self.session_data),
            "heart_rate_data": [data.model_dump() for data in self.session_data],
            "hrv_analysis": self.analyze_session().model_dump() if self.analyze_session() else None
        }
        
        self.sensor.save_data(session_summary, file_path)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sensor.disconnect()

# Example usage
if __name__ == "__main__":
    # Create user
    user = User(name="John Doe", age=35, gender="male")
    
    # Scan for devices first
    sensor = HeartRateSensor()
    devices = sensor.scan_devices()
    logger.info(f"Found devices: {devices}")
    
    # Start monitoring session
    with BiometricSession(user, sensor_mac="00:11:22:33:44:55") as session:
        session.collect_data(duration=30, interval=1.0)
        session.save_session()