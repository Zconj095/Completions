import json
import time
import subprocess
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HeartRateData:
    """Data class for heart rate measurements"""
    heart_rate: float
    timestamp: float
    quality: Optional[str] = None

class HeartRateSensor:
    """Enhanced heart rate sensor class with better error handling"""
    
    def __init__(self, sensor_mac: Optional[str] = None):
        self.sensor_mac = sensor_mac
        self.is_connected = False
        logger.info(f"Initializing sensor with MAC: {sensor_mac}")
        
    def connect(self) -> bool:
        """Connect to the heart rate sensor"""
        try:
            # Simulate sensor connection (replace with actual Bluetooth logic)
            logger.info("Attempting to connect to heart rate sensor...")
            time.sleep(1)  # Simulate connection delay
            self.is_connected = True
            logger.info("Successfully connected to sensor")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to sensor: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the sensor"""
        self.is_connected = False
        logger.info("Disconnected from sensor")
    
    def get_data(self) -> Optional[HeartRateData]:
        """Get heart rate data from sensor"""
        if not self.is_connected:
            logger.warning("Sensor not connected")
            return None
            
        try:
            # Simulate reading data (replace with actual sensor reading)
            import random
            bpm = random.randint(60, 100)  # Simulate heart rate
            timestamp = time.time()
            quality = "good" if bpm > 65 else "fair"
            
            return HeartRateData(
                heart_rate=bpm,
                timestamp=timestamp,
                quality=quality
            )
        except Exception as e:
            logger.error(f"Error reading sensor data: {e}")
            return None
    
    def save_data(self, data: List[HeartRateData], file_path: str):
        """Save heart rate data to JSON file"""
        try:
            data_dict = [
                {
                    "heart_rate": d.heart_rate,
                    "timestamp": d.timestamp,
                    "quality": d.quality
                } for d in data
            ]
            
            with open(file_path, 'w') as file:
                json.dump(data_dict, file, indent=2)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

class MegaStorage:
    """Enhanced Mega.nz storage handler"""
    
    @staticmethod
    def upload_to_mega(file_path: str, remote_path: str) -> bool:
        """Upload file to Mega.nz"""
        try:
            if not Path(file_path).exists():
                logger.error(f"File {file_path} does not exist")
                return False
                
            logger.info(f"Uploading {file_path} to {remote_path}")
            # Use megacmd if available
            result = subprocess.run(
                ["megaput", file_path, remote_path],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Upload successful")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("megacmd not found. Please install MEGAcmd")
            return False
    
    @staticmethod
    def download_from_mega(remote_path: str, local_path: str) -> bool:
        """Download file from Mega.nz"""
        try:
            logger.info(f"Downloading {remote_path} to {local_path}")
            result = subprocess.run(
                ["megaget", remote_path, local_path],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Download successful")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("megacmd not found. Please install MEGAcmd")
            return False

class HRVAnalyzer:
    """Heart Rate Variability analyzer"""
    
    @staticmethod
    def read_data_from_file(file_path: str) -> Optional[List[Dict]]:
        """Read heart rate data from JSON file"""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            logger.info(f"Loaded {len(data)} data points from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error reading data file: {e}")
            return None
    
    @staticmethod
    def calculate_basic_hrv(heart_rates: List[float]) -> Dict[str, float]:
        """Calculate basic HRV metrics without neurokit2"""
        if len(heart_rates) < 2:
            return {}
        
        # Calculate RR intervals (simplified)
        rr_intervals = []
        for i in range(1, len(heart_rates)):
            if heart_rates[i] > 0:
                rr = 60000 / heart_rates[i]  # Convert BPM to RR interval in ms
                rr_intervals.append(rr)
        
        if len(rr_intervals) < 2:
            return {}
        
        # Calculate RMSSD (Root Mean Square of Successive Differences)
        diff_squares = [(rr_intervals[i+1] - rr_intervals[i])**2 
                       for i in range(len(rr_intervals)-1)]
        rmssd = (sum(diff_squares) / len(diff_squares)) ** 0.5
        
        # Calculate pNN50 (percentage of intervals differing by >50ms)
        nn50_count = sum(1 for i in range(len(rr_intervals)-1) 
                        if abs(rr_intervals[i+1] - rr_intervals[i]) > 50)
        pnn50 = (nn50_count / (len(rr_intervals) - 1)) * 100
        
        return {
            "rmssd": round(rmssd, 2),
            "pnn50": round(pnn50, 2),
            "mean_hr": round(sum(heart_rates) / len(heart_rates), 2),
            "hr_std": round((sum([(hr - sum(heart_rates)/len(heart_rates))**2 
                                for hr in heart_rates]) / len(heart_rates)) ** 0.5, 2)
        }

class HeartRateMonitor:
    """Main application class"""
    
    def __init__(self, sensor_mac: Optional[str] = None):
        self.sensor = HeartRateSensor(sensor_mac)
        self.storage = MegaStorage()
        self.analyzer = HRVAnalyzer()
        self.data_buffer: List[HeartRateData] = []
    
    def collect_data(self, duration_seconds: int = 60, interval_seconds: int = 1):
        """Collect heart rate data for specified duration"""
        if not self.sensor.connect():
            return False
        
        logger.info(f"Starting data collection for {duration_seconds} seconds")
        end_time = time.time() + duration_seconds
        
        try:
            while time.time() < end_time:
                data = self.sensor.get_data()
                if data:
                    self.data_buffer.append(data)
                    logger.info(f"HR: {data.heart_rate} BPM, Quality: {data.quality}")
                
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        finally:
            self.sensor.disconnect()
        
        logger.info(f"Collected {len(self.data_buffer)} data points")
        return True
    
    def analyze_and_save(self, local_file: str = "heart_rate_data.json"):
        """Analyze collected data and save results"""
        if not self.data_buffer:
            logger.warning("No data to analyze")
            return None
        
        # Save raw data
        self.sensor.save_data(self.data_buffer, local_file)
        
        # Perform HRV analysis
        heart_rates = [d.heart_rate for d in self.data_buffer]
        hrv_metrics = self.analyzer.calculate_basic_hrv(heart_rates)
        
        # Save analysis results
        analysis_file = local_file.replace('.json', '_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(hrv_metrics, f, indent=2)
        
        logger.info(f"HRV Analysis Results: {hrv_metrics}")
        return hrv_metrics
    
    def sync_to_cloud(self, local_file: str, remote_path: str):
        """Upload data to Mega.nz"""
        return self.storage.upload_to_mega(local_file, remote_path)

def main():
    """Enhanced main function with better workflow"""
    try:
        # Initialize monitor
        monitor = HeartRateMonitor()
        
        # Collect data
        if monitor.collect_data(duration_seconds=30, interval_seconds=2):
            # Analyze and save
            results = monitor.analyze_and_save("sensor_data.json")
            
            if results:
                # Upload to cloud storage
                success = monitor.sync_to_cloud(
                    "sensor_data.json", 
                    "/MySensorData/sensor_data.json"
                )
                
                if success:
                    logger.info("Data successfully synced to cloud")
                else:
                    logger.warning("Cloud sync failed, data saved locally")
        
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
