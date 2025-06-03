# sensors/heart_rate.py
import time
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import threading
import queue
import random
import numpy as np
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartRateSensor:
    def __init__(self, sensor_mac: Optional[str] = None): 
        self.sensor_mac = sensor_mac
        self.connected = False
        self.last_reading = None
        self.reading_history = []
        self.max_history = 1000
        self.baseline_hr = random.randint(60, 80)  # Individual baseline
        
    def connect(self, sensor_mac: str) -> bool:
        """Connect to Bluetooth sensor with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to sensor {sensor_mac} (attempt {attempt + 1})")
                time.sleep(0.5)
                self.sensor_mac = sensor_mac
                self.connected = True
                logger.info("Successfully connected to heart rate sensor")
                return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return False

    def disconnect(self):
        """Disconnect from sensor"""
        self.connected = False
        logger.info("Disconnected from heart rate sensor")

    def get_data(self) -> Optional[Dict[str, Any]]:
        if not self.connected:
            return None
            
        try:
            # More realistic HR simulation with circadian rhythm
            hour = datetime.now().hour
            circadian_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Activity-based variation
            activity_factor = random.choice([1.0, 1.2, 1.5, 0.9])  # rest, light, moderate, recovery
            
            base_bpm = self.baseline_hr * circadian_factor * activity_factor
            variation = random.gauss(0, 5)
            bpm = max(45, min(180, int(base_bpm + variation)))
            
            timestamp = time.time()
            quality = "excellent" if abs(variation) < 3 else "good" if abs(variation) < 8 else "fair"
            
            # Realistic RR interval with some variation
            base_rr = 60000 / bpm if bpm > 0 else 1000
            rr_variation = random.gauss(0, base_rr * 0.05)  # 5% variation
            rr_interval = base_rr + rr_variation
            
            reading = {
                "heart_rate": bpm,
                "timestamp": timestamp,
                "quality": quality,
                "sensor_mac": self.sensor_mac,
                "rr_interval": rr_interval,
                "activity_level": self._estimate_activity_level(bpm)
            }
            
            self.last_reading = reading
            self._update_history(reading)
            return reading
            
        except Exception as e:
            logger.error(f"Error reading sensor data: {e}")
            return None

    def _estimate_activity_level(self, hr: int) -> str:
        """Estimate activity level based on heart rate"""
        if hr < self.baseline_hr + 10:
            return "resting"
        elif hr < self.baseline_hr + 30:
            return "light"
        elif hr < self.baseline_hr + 50:
            return "moderate"
        else:
            return "vigorous"

    def _update_history(self, reading: Dict[str, Any]):
        """Maintain a rolling history of readings"""
        self.reading_history.append(reading)
        if len(self.reading_history) > self.max_history:
            self.reading_history.pop(0)

    def get_recent_readings(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get readings from the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        return [r for r in self.reading_history if r["timestamp"] >= cutoff_time]

# Enhanced SleepTracker with realistic data
class SleepTracker:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.sleep_patterns = {
            "bedtime_avg": 23.0,  # 11 PM
            "wake_time_avg": 7.0,  # 7 AM
            "efficiency": 0.85
        }
    
    def get_sleep_data(self, date: str = None) -> Dict[str, Any]:
        """Generate realistic sleep data"""
        if not date:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Simulate realistic sleep patterns
        bedtime_variation = random.gauss(0, 0.5)  # ±30 min variation
        duration_variation = random.gauss(0, 30)  # ±30 min variation
        
        total_duration = 480 + duration_variation  # ~8 hours base
        sleep_efficiency = self.sleep_patterns["efficiency"] + random.gauss(0, 0.1)
        asleep_time = total_duration * sleep_efficiency
        
        # Sleep stage distribution (percentages of asleep time)
        deep_sleep_pct = random.uniform(0.15, 0.25)
        rem_sleep_pct = random.uniform(0.20, 0.25)
        light_sleep_pct = 1 - deep_sleep_pct - rem_sleep_pct
        
        deep_sleep = asleep_time * deep_sleep_pct
        rem_sleep = asleep_time * rem_sleep_pct
        light_sleep = asleep_time * light_sleep_pct
        
        # Calculate sleep score based on multiple factors
        duration_score = min(100, (asleep_time / 480) * 100)
        efficiency_score = sleep_efficiency * 100
        deep_sleep_score = min(100, (deep_sleep / (480 * 0.2)) * 100)
        
        sleep_score = (duration_score * 0.4 + efficiency_score * 0.4 + deep_sleep_score * 0.2)
        
        return {
            'user_id': self.user_id,
            'date': date,
            'duration': int(total_duration),
            'asleep_time': int(asleep_time),
            'deep_sleep': int(deep_sleep),
            'rem_sleep': int(rem_sleep),
            'light_sleep': int(light_sleep),
            'sleep_score': round(sleep_score, 1),
            'sleep_efficiency': round(sleep_efficiency * 100, 1),
            'wake_ups': random.randint(0, 4)
        }

# Enhanced LocationTracker with realistic locations
class LocationTracker:
    def __init__(self):
        self.locations = {
            'home': {'lat': 40.7128, 'lon': -74.0060, 'activities': ['sleeping', 'resting', 'working']},
            'office': {'lat': 40.7589, 'lon': -73.9851, 'activities': ['working', 'meeting']},
            'gym': {'lat': 40.7505, 'lon': -73.9934, 'activities': ['exercising', 'training']},
            'park': {'lat': 40.7829, 'lon': -73.9654, 'activities': ['walking', 'running', 'relaxing']},
            'restaurant': {'lat': 40.7614, 'lon': -73.9776, 'activities': ['eating', 'socializing']}
        }
        self.current_location = 'home'
    
    def get_location(self) -> Dict[str, Any]:
        """Get current location with realistic context"""
        # Simulate location changes based on time of day
        hour = datetime.now().hour
        if 6 <= hour < 9:
            self.current_location = random.choice(['home', 'gym'])
        elif 9 <= hour < 17:
            self.current_location = 'office'
        elif 17 <= hour < 20:
            self.current_location = random.choice(['office', 'gym', 'restaurant'])
        else:
            self.current_location = random.choice(['home', 'restaurant'])
        
        loc_data = self.locations[self.current_location]
        activity = random.choice(loc_data['activities'])
        
        return {
            'location_name': self.current_location,
            'latitude': loc_data['lat'] + random.gauss(0, 0.001),
            'longitude': loc_data['lon'] + random.gauss(0, 0.001),
            'timestamp': time.time(),
            'activity': activity,
            'confidence': random.uniform(0.8, 1.0)
        }

# Enhanced database operations
import sqlite3
from typing import Dict, Any, List, Optional

class BioDatabase:
    def __init__(self, db_path: str = "bio_data.db"):
        self.db_path = db_path
        self._init_database()
        self._populate_sample_data()

    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                password_hash TEXT
            )
        ''')
        
        # Heart rate readings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS heart_rate_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                heart_rate INTEGER,
                timestamp REAL,
                quality TEXT,
                sensor_mac TEXT,
                rr_interval REAL,
                activity_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Sleep data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                date TEXT,
                duration INTEGER,
                asleep_time INTEGER,
                deep_sleep INTEGER,
                rem_sleep INTEGER,
                light_sleep INTEGER,
                sleep_score REAL,
                sleep_efficiency REAL,
                wake_ups INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Location data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS location_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                location_name TEXT,
                latitude REAL,
                longitude REAL,
                timestamp REAL,
                activity TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _populate_sample_data(self):
        """Populate database with sample data for testing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if sample data already exists
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Create sample user
        sample_user_id = "sample-user-123"
        cursor.execute('''
            INSERT OR IGNORE INTO users (id, name, age, gender, email)
            VALUES (?, ?, ?, ?, ?)
        ''', (sample_user_id, "Sample User", 30, "male", "sample@example.com"))
        
        # Generate sample heart rate data for the past week
        now = time.time()
        for i in range(7 * 24 * 6):  # 7 days, every 10 minutes
            timestamp = now - (i * 600)  # 10 minutes ago
            hr = random.randint(60, 100)
            rr_interval = 60000 / hr + random.gauss(0, 50)
            
            cursor.execute('''
                INSERT INTO heart_rate_readings 
                (user_id, heart_rate, timestamp, quality, sensor_mac, rr_interval, activity_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (sample_user_id, hr, timestamp, "good", "sample:sensor", rr_interval, "resting"))
        
        # Generate sample sleep data for the past week
        for i in range(7):
            sleep_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            duration = random.randint(420, 540)  # 7-9 hours
            asleep_time = duration * random.uniform(0.8, 0.95)
            deep_sleep = asleep_time * random.uniform(0.15, 0.25)
            rem_sleep = asleep_time * random.uniform(0.20, 0.25)
            light_sleep = asleep_time - deep_sleep - rem_sleep
            sleep_score = random.uniform(70, 95)
            
            cursor.execute('''
                INSERT INTO sleep_data 
                (user_id, date, duration, asleep_time, deep_sleep, rem_sleep, light_sleep, sleep_score, sleep_efficiency, wake_ups)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (sample_user_id, sleep_date, duration, asleep_time, deep_sleep, rem_sleep, light_sleep, sleep_score, 85.0, random.randint(1, 3)))
        
        conn.commit()
        conn.close()
        logger.info("Sample data populated successfully")

    def insert_reading(self, table: str, data: Dict[str, Any]):
        """Insert reading into specified table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if table == "heart_rate":
                cursor.execute('''
                    INSERT INTO heart_rate_readings 
                    (user_id, heart_rate, timestamp, quality, sensor_mac, rr_interval, activity_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('user_id'),
                    data.get('heart_rate'),
                    data.get('timestamp'),
                    data.get('quality'),
                    data.get('sensor_mac'),
                    data.get('rr_interval'),
                    data.get('activity_level')
                ))
            elif table == "sleep_data":
                cursor.execute('''
                    INSERT INTO sleep_data 
                    (user_id, date, duration, asleep_time, deep_sleep, rem_sleep, light_sleep, sleep_score, sleep_efficiency, wake_ups)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('user_id'),
                    data.get('date'),
                    data.get('duration'),
                    data.get('asleep_time'),
                    data.get('deep_sleep'),
                    data.get('rem_sleep'),
                    data.get('light_sleep'),
                    data.get('sleep_score'),
                    data.get('sleep_efficiency'),
                    data.get('wake_ups')
                ))
            # ... other table handlers
            
            conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_user_readings(self, user_id: str, table: str = "heart_rate_readings", 
                         limit: int = 100, start_date: Optional[datetime] = None) -> List[Dict]:
        """Get readings for a specific user with optional date filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = f"SELECT * FROM {table} WHERE user_id = ?"
            params = [user_id]
            
            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date.isoformat())
            
            query += f" ORDER BY created_at DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving data: {e}")
            return []
        finally:
            conn.close()

    def get_user_stats(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Heart rate stats
            cursor.execute('''
                SELECT AVG(heart_rate) as avg_hr, MIN(heart_rate) as min_hr, 
                       MAX(heart_rate) as max_hr, COUNT(*) as reading_count,
                       AVG(rr_interval) as avg_rr
                FROM heart_rate_readings 
                WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
            '''.format(days), (user_id,))
            
            hr_stats = cursor.fetchone()
            
            # Sleep stats
            cursor.execute('''
                SELECT AVG(duration) as avg_duration, AVG(sleep_score) as avg_score,
                       AVG(deep_sleep) as avg_deep, AVG(sleep_efficiency) as avg_efficiency,
                       AVG(wake_ups) as avg_wake_ups, COUNT(*) as sleep_count
                FROM sleep_data 
                WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
            '''.format(days), (user_id,))
            
            sleep_stats = cursor.fetchone()
            
            # Activity distribution
            cursor.execute('''
                SELECT activity_level, COUNT(*) as count
                FROM heart_rate_readings 
                WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
                GROUP BY activity_level
            '''.format(days), (user_id,))
            
            activity_data = cursor.fetchall()
            activity_distribution = {row[0]: row[1] for row in activity_data}
            
            return {
                "heart_rate": {
                    "average": round(hr_stats[0], 1) if hr_stats[0] else 0,
                    "minimum": hr_stats[1] if hr_stats[1] else 0,
                    "maximum": hr_stats[2] if hr_stats[2] else 0,
                    "reading_count": hr_stats[3] if hr_stats[3] else 0,
                    "avg_rr_interval": round(hr_stats[4], 1) if hr_stats[4] else 0
                },
                "sleep": {
                    "average_duration": round(sleep_stats[0], 1) if sleep_stats[0] else 0,
                    "average_score": round(sleep_stats[1], 1) if sleep_stats[1] else 0,
                    "average_deep_sleep": round(sleep_stats[2], 1) if sleep_stats[2] else 0,
                    "average_efficiency": round(sleep_stats[3], 1) if sleep_stats[3] else 0,
                    "average_wake_ups": round(sleep_stats[4], 1) if sleep_stats[4] else 0,
                    "night_count": sleep_stats[5] if sleep_stats[5] else 0
                },
                "activity": activity_distribution
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving stats: {e}")
            return {}
        finally:
            conn.close()

# Enhanced analytics
def extract_hrv(rr_intervals: List[float]) -> Dict[str, float]:
    """Extract comprehensive HRV features from RR intervals"""
    if not rr_intervals or len(rr_intervals) < 10:
        return {"rmssd": 0.0, "pnn50": 0.0, "sdnn": 0.0, "lf_hf_ratio": 0.0, "stress_index": 0.0}
    
    rr_array = np.array(rr_intervals)
    
    # Time domain measures
    sdnn = np.std(rr_array)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_array))))
    
    # pNN50
    nn50 = np.sum(np.abs(np.diff(rr_array)) > 50)
    pnn50 = (nn50 / len(rr_array)) * 100 if len(rr_array) > 0 else 0
    
    # Frequency domain analysis
    if len(rr_array) > 50:
        t = np.cumsum(rr_array)
        t_interp = np.arange(0, t[-1], 1000)
        rr_interp = np.interp(t_interp, t, rr_array)
        
        freqs, psd = signal.welch(rr_interp, fs=1.0, nperseg=min(256, len(rr_interp)//4))
        
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    else:
        lf_hf_ratio = 1.0
    
    # Improved stress index calculation
    stress_index = max(0, min(100, 100 - (rmssd * 1.5) - (pnn50 * 0.5)))
    
    return {
        "rmssd": round(float(rmssd), 2),
        "pnn50": round(float(pnn50), 2),
        "sdnn": round(float(sdnn), 2),
        "lf_hf_ratio": round(float(lf_hf_ratio), 2),
        "stress_index": round(float(stress_index), 1)
    }

# UserManager class
class UserManager:
    def __init__(self, db: BioDatabase):
        self.db = db
    
    def register(self, user_data: Dict[str, Any]) -> str:
        """Register a new user"""
        import uuid
        user_id = str(uuid.uuid4())
        user_data['id'] = user_id
        self.db.insert_reading("users", user_data)
        return user_id

# Enhanced User model with validation
class User:
    def __init__(self, name: str, age: int, gender: str, user_id: Optional[str] = None, email: Optional[str] = None):
        self.id = user_id
        self.name = self._validate_name(name)
        self.age = self._validate_age(age)
        self.gender = self._validate_gender(gender)
        self.email = self._validate_email(email) if email else None

    def _validate_name(self, name: str) -> str:
        if not name or len(name.strip()) < 2:
            raise ValueError("Name must be at least 2 characters long")
        return name.strip()

    def _validate_age(self, age: int) -> int:
        if not isinstance(age, int) or age < 0 or age > 150:
            raise ValueError("Age must be a valid integer between 0 and 150")
        return age

    def _validate_gender(self, gender: str) -> str:
        valid_genders = ["male", "female", "other", "prefer_not_to_say"]
        if gender.lower() not in valid_genders:
            raise ValueError(f"Gender must be one of: {', '.join(valid_genders)}")
        return gender.lower()

    def _validate_email(self, email: str) -> str:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "email": self.email
        }

# Enhanced monitoring system
class EnhancedBioHealthApp:
    def __init__(self):
        self.db = BioDatabase()
        self.user_manager = UserManager(self.db)
        self.current_user = None
        self.monitoring_active = False
        self.data_queue = queue.Queue()
        self.analysis_thread = None
        
    def generate_health_report(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        stats = self.db.get_user_stats(user_id, days)
        
        # Get recent HR data for HRV analysis
        hr_readings = self.db.get_user_readings(user_id, "heart_rate_readings", limit=500)
        rr_intervals = [r.get('rr_interval') for r in hr_readings if r.get('rr_interval')]
        
        hrv_metrics = extract_hrv(rr_intervals) if rr_intervals else {}
        
        # Calculate wellness score
        wellness_score = self._calculate_wellness_score(stats, hrv_metrics)
        
        # Get sleep trends
        sleep_trend = self._analyze_sleep_trends(user_id, days)
        
        return {
            "user_id": user_id,
            "report_period_days": days,
            "generated_at": datetime.now().isoformat(),
            "wellness_score": wellness_score,
            "statistics": stats,
            "hrv_analysis": hrv_metrics,
            "sleep_trends": sleep_trend,
            "recommendations": self._generate_enhanced_recommendations(stats, hrv_metrics, wellness_score),
            "insights": self._generate_insights(stats, hrv_metrics)
        }

    def _calculate_wellness_score(self, stats: Dict, hrv_metrics: Dict) -> Dict[str, float]:
        """Calculate overall wellness score"""
        scores = {}
        
        # Heart rate score
        hr_stats = stats.get("heart_rate", {})
        avg_hr = hr_stats.get("average", 0)
        if 60 <= avg_hr <= 100:
            scores["heart_rate"] = 100 - abs(avg_hr - 70) * 2
        else:
            scores["heart_rate"] = max(0, 100 - abs(avg_hr - 70) * 3)
        
        # Sleep score
        sleep_stats = stats.get("sleep", {})
        sleep_score = sleep_stats.get("average_score", 0)
        scores["sleep"] = sleep_score
        
        # HRV score
        rmssd = hrv_metrics.get("rmssd", 0)
        if rmssd > 30:
            scores["hrv"] = 90
        elif rmssd > 20:
            scores["hrv"] = 70
        else:
            scores["hrv"] = 50
        
        # Activity score based on distribution
        activity_dist = stats.get("activity", {})
        total_readings = sum(activity_dist.values()) if activity_dist else 1
        active_readings = activity_dist.get("moderate", 0) + activity_dist.get("vigorous", 0)
        activity_ratio = active_readings / total_readings if total_readings > 0 else 0
        scores["activity"] = min(100, activity_ratio * 500)  # 20% active time = 100 score
        
        # Overall score
        overall = sum(scores.values()) / len(scores) if scores else 0
        scores["overall"] = round(overall, 1)
        
        return {k: round(v, 1) for k, v in scores.items()}

    def _analyze_sleep_trends(self, user_id: str, days: int) -> Dict[str, Any]:
        """Analyze sleep trends over time"""
        sleep_data = self.db.get_user_readings(user_id, "sleep_data", limit=days)
        
        if not sleep_data:
            return {"trend": "no_data", "consistency": 0}
        
        scores = [d.get("sleep_score", 0) for d in sleep_data]
        durations = [d.get("duration", 0) for d in sleep_data]
        
        # Calculate trends
        if len(scores) > 3:
            recent_avg = np.mean(scores[:len(scores)//2])
            older_avg = np.mean(scores[len(scores)//2:])
            
            if recent_avg > older_avg + 5:
                trend = "improving"
            elif recent_avg < older_avg - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Calculate consistency
        consistency = 100 - (np.std(scores) * 2) if scores else 0
        consistency = max(0, min(100, consistency))
        
        return {
            "trend": trend,
            "consistency": round(consistency, 1),
            "average_duration": round(np.mean(durations), 1) if durations else 0,
            "duration_variability": round(np.std(durations), 1) if durations else 0
        }

    def _generate_enhanced_recommendations(self, stats: Dict, hrv_metrics: Dict, wellness_score: Dict) -> List[Dict[str, str]]:
        """Generate detailed, actionable recommendations"""
        recommendations = []
        
        # Heart rate recommendations
        hr_stats = stats.get("heart_rate", {})
        avg_hr = hr_stats.get("average", 0)
        
        if avg_hr > 100:
            recommendations.append({
                "category": "cardiovascular",
                "priority": "high",
                "message": "Your resting heart rate is elevated. Consider stress management techniques and consult a healthcare provider.",
                "action": "Practice deep breathing exercises 10 minutes daily"
            })
        elif avg_hr < 50 and wellness_score.get("activity", 0) < 70:
            recommendations.append({
                "category": "cardiovascular",
                "priority": "medium",
                "message": "Very low heart rate detected. If you're not an athlete, consider medical evaluation.",
                "action": "Schedule a check-up with your doctor"
            })
        
        # Sleep recommendations
        sleep_stats = stats.get("sleep", {})
        avg_duration = sleep_stats.get("average_duration", 0)
        avg_score = sleep_stats.get("average_score", 0)
        
        if avg_duration < 420:  # Less than 7 hours
            recommendations.append({
                "category": "sleep",
                "priority": "high",
                "message": "You're not getting enough sleep. Aim for 7-9 hours per night.",
                "action": "Set a consistent bedtime 30 minutes earlier"
            })
        
        if avg_score < 70:
            recommendations.append({
                "category": "sleep",
                "priority": "medium",
                "message": "Your sleep quality could be improved.",
                "action": "Avoid screens 1 hour before bed and keep your room cool"
            })
        
        # HRV recommendations
        stress_index = hrv_metrics.get("stress_index", 50)
        if stress_index > 70:
            recommendations.append({
                "category": "stress",
                "priority": "high",
                "message": "High stress levels detected based on heart rate variability.",
                "action": "Try meditation, yoga, or other relaxation techniques"
            })
        
        # Activity recommendations
        activity_score = wellness_score.get("activity", 0)
        if activity_score < 50:
            recommendations.append({
                "category": "activity",
                "priority": "medium",
                "message": "Increase your physical activity levels for better health.",
                "action": "Aim for 30 minutes of moderate exercise 5 days per week"
            })
        
        # Overall wellness
        overall_score = wellness_score.get("overall", 0)
        if overall_score > 80:
            recommendations.append({
                "category": "general",
                "priority": "low",
                "message": "Excellent health metrics! Keep up the great work.",
                "action": "Maintain your current healthy lifestyle"
            })
        
        return recommendations

    def _generate_insights(self, stats: Dict, hrv_metrics: Dict) -> List[str]:
        """Generate personalized health insights"""
        insights = []
        
        hr_stats = stats.get("heart_rate", {})
        sleep_stats = stats.get("sleep", {})
        
        # Heart rate insights
        if hr_stats.get("reading_count", 0) > 0:
            hr_range = hr_stats.get("maximum", 0) - hr_stats.get("minimum", 0)
            if hr_range > 60:
                insights.append("Your heart rate shows good variability, indicating healthy cardiovascular adaptation.")
            
        # Sleep insights
        if sleep_stats.get("average_efficiency", 0) > 85:
            insights.append("Your sleep efficiency is excellent - you fall asleep quickly and stay asleep.")
        
        # HRV insights
        rmssd = hrv_metrics.get("rmssd", 0)
        if rmssd > 40:
            insights.append("Your heart rate variability suggests excellent recovery and low stress levels.")
        
        # Pattern recognition
        activity_dist = stats.get("activity", {})
        if activity_dist:
            most_common_activity = max(activity_dist, key=activity_dist.get)
            insights.append(f"You spend most of your time in '{most_common_activity}' activity level.")
        
        return insights

# Usage example with enhanced features
if __name__ == "__main__":
    app = EnhancedBioHealthApp()
    
    # Use the sample user that was auto-created
    sample_user_id = "sample-user-123"
    
    # Generate and display enhanced health report
    report = app.generate_health_report(sample_user_id, days=7)
    print(json.dumps(report, indent=2))
