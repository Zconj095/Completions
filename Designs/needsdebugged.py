import datetime
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Define enums for better type safety
class Season(Enum):
    SPRING = "Spring"
    SUMMER = "Summer"
    FALL = "Fall"
    WINTER = "Winter"

class MoonPhase(Enum):
    NEW = "New"
    WAXING_CRESCENT = "Waxing Crescent"
    FIRST_QUARTER = "First Quarter"
    WAXING_GIBBOUS = "Waxing Gibbous"
    FULL = "Full"
    WANING_GIBBOUS = "Waning Gibbous"
    LAST_QUARTER = "Last Quarter"
    WANING_CRESCENT = "Waning Crescent"

class CircadianType(Enum):
    NIGHT_OWL = "Night Owl"
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"

# Data classes for better structure
@dataclass
class BiometricData:
    glucose: Optional[float] = None
    weight: Optional[float] = None
    blood_pressure: Optional[Tuple[int, int]] = None
    heart_rate: Optional[int] = None
    respiration_rate: Optional[int] = None

@dataclass
class HormoneProfile:
    leptin: float = 0.0
    ghrelin: float = 0.0
    cortisol: float = 0.0
    serotonin: float = 0.0
    dopamine: float = 0.0
    oxytocin: float = 0.0

@dataclass
class Person:
    name: str
    birth_date: datetime.date
    birth_time: datetime.time
    location: Tuple[float, float]  # (latitude, longitude)
    blood_type: Optional[str] = None
    bmi: Optional[float] = None

# Constants and configurations
SEASONS_DATA = {
    Season.SPRING: {
        'energy': 8,
        'mood': 7,
        'health': 8,
        'description': 'Time of renewal and growth'
    },
    Season.SUMMER: {
        'energy': 10,
        'mood': 8,
        'health': 9,
        'description': 'Peak physical energy and vitality'
    },
    Season.FALL: {
        'energy': 6,
        'mood': 6,
        'health': 7,
        'description': 'Time of harvest and preparation'
    },
    Season.WINTER: {
        'energy': 4,
        'mood': 5,
        'health': 6,
        'description': 'Time of rest and introspection'
    }
}

MOON_PHASES_DATA = {
    MoonPhase.FULL: {
        'influence': 'climax of energy',
        'advice': 'harness intensity for important tasks'
    },
    MoonPhase.NEW: {
        'influence': 'fresh starts',
        'advice': 'set intentions and begin new projects'
    },
    MoonPhase.WAXING_CRESCENT: {
        'influence': 'building energy',
        'advice': 'take action on your intentions'
    },
    MoonPhase.WANING_CRESCENT: {
        'influence': 'releasing energy',
        'advice': 'let go and prepare for renewal'
    }
}

CIRCADIAN_ADVICE = {
    CircadianType.NIGHT_OWL: "Schedule important tasks for evening hours",
    CircadianType.MORNING: "Take advantage of early morning energy",
    CircadianType.AFTERNOON: "Peak performance in afternoon hours",
    CircadianType.EVENING: "Evening focus and creativity time"
}

HORMONE_BASELINES = {
    "leptin": {
        Season.SPRING: 85,
        Season.SUMMER: 80,
        Season.FALL: 95,
        Season.WINTER: 100
    },
    "ghrelin": {
        Season.SPRING: 10,
        Season.SUMMER: 15,
        Season.FALL: 5,
        Season.WINTER: 8
    },
    "cortisol": {
        Season.SPRING: 25,
        Season.SUMMER: 30,
        Season.FALL: 20,
        Season.WINTER: 15
    }
}

class BioCycleCalculator:
    """Main class for calculating biological cycles and patterns"""
    
    def __init__(self, person: Person):
        self.person = person
    
    def get_season(self, date: datetime.date) -> Season:
        """Determine season based on date"""
        month = date.month
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.FALL
        else:
            return Season.WINTER
    
    def get_moon_phase(self, date: datetime.date) -> MoonPhase:
        """Calculate moon phase (simplified calculation)"""
        # Simplified moon phase calculation
        days_since_new = (date - datetime.date(2000, 1, 6)).days % 29.53
        
        if days_since_new < 1:
            return MoonPhase.NEW
        elif days_since_new < 7:
            return MoonPhase.WAXING_CRESCENT
        elif days_since_new < 9:
            return MoonPhase.FIRST_QUARTER
        elif days_since_new < 14:
            return MoonPhase.WAXING_GIBBOUS
        elif days_since_new < 16:
            return MoonPhase.FULL
        elif days_since_new < 22:
            return MoonPhase.WANING_GIBBOUS
        elif days_since_new < 24:
            return MoonPhase.LAST_QUARTER
        else:
            return MoonPhase.WANING_CRESCENT
    
    def get_circadian_tendency(self, birth_time: datetime.time) -> CircadianType:
        """Determine circadian type based on birth time"""
        hour = birth_time.hour
        if hour < 6:
            return CircadianType.NIGHT_OWL
        elif hour < 12:
            return CircadianType.MORNING
        elif hour < 18:
            return CircadianType.AFTERNOON
        else:
            return CircadianType.EVENING
    
    def calculate_biorhythms(self, date: datetime.date) -> Dict:
        """Calculate all biorhythms for a given date"""
        season = self.get_season(date)
        moon = self.get_moon_phase(date)
        circadian = self.get_circadian_tendency(self.person.birth_time)
        
        # Calculate days since birth for biorhythm cycles
        days_alive = (date - self.person.birth_date).days
        
        # Standard biorhythm cycles (23, 28, 33 days)
        physical = math.sin(2 * math.pi * days_alive / 23)
        emotional = math.sin(2 * math.pi * days_alive / 28)
        intellectual = math.sin(2 * math.pi * days_alive / 33)
        
        return {
            'date': date,
            'season': season,
            'moon_phase': moon,
            'circadian_type': circadian,
            'physical_cycle': round(physical, 3),
            'emotional_cycle': round(emotional, 3),
            'intellectual_cycle': round(intellectual, 3)
        }

class HormoneCalculator:
    """Calculate hormone levels based on various factors"""
    
    def __init__(self, person: Person):
        self.person = person
    
    def calculate_seasonal_hormones(self, date: datetime.date) -> HormoneProfile:
        """Calculate hormone levels based on season"""
        season = self._get_season(date)
        
        leptin = HORMONE_BASELINES["leptin"][season]
        ghrelin = HORMONE_BASELINES["ghrelin"][season]
        cortisol = HORMONE_BASELINES["cortisol"][season]
        
        # Apply BMI modifier if available
        if self.person.bmi:
            leptin_resistance = self._calculate_leptin_resistance(self.person.bmi)
            leptin *= (1 - leptin_resistance)
        
        return HormoneProfile(
            leptin=leptin,
            ghrelin=ghrelin,
            cortisol=cortisol
        )
    
    def _get_season(self, date: datetime.date) -> Season:
        """Helper method to get season"""
        month = date.month
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.FALL
        else:
            return Season.WINTER
    
    def _calculate_leptin_resistance(self, bmi: float) -> float:
        """Calculate leptin resistance factor based on BMI"""
        if bmi > 30:
            return 0.3  # 30% resistance
        elif bmi > 25:
            return 0.1  # 10% resistance
        else:
            return 0.0  # No resistance

class AdviceGenerator:
    """Generate personalized advice based on cycles"""
    
    @staticmethod
    def generate_advice(biorhythms: Dict) -> str:
        """Generate advice based on current biorhythm state"""
        advice_parts = []
        
        # Season advice
        season_data = SEASONS_DATA[biorhythms['season']]
        advice_parts.append(f"Season: {season_data['description']}")
        
        # Moon phase advice
        moon_data = MOON_PHASES_DATA.get(biorhythms['moon_phase'])
        if moon_data:
            advice_parts.append(f"Moon: {moon_data['advice']}")
        
        # Circadian advice
        circadian_advice = CIRCADIAN_ADVICE.get(biorhythms['circadian_type'])
        if circadian_advice:
            advice_parts.append(f"Timing: {circadian_advice}")
        
        # Biorhythm cycle advice
        if biorhythms['physical_cycle'] > 0.5:
            advice_parts.append("Physical: High energy period - good for exercise")
        elif biorhythms['physical_cycle'] < -0.5:
            advice_parts.append("Physical: Low energy period - focus on rest")
        
        if biorhythms['emotional_cycle'] > 0.5:
            advice_parts.append("Emotional: Positive emotional state - good for social activities")
        elif biorhythms['emotional_cycle'] < -0.5:
            advice_parts.append("Emotional: Sensitive period - practice self-care")
        
        if biorhythms['intellectual_cycle'] > 0.5:
            advice_parts.append("Mental: Sharp thinking period - tackle complex tasks")
        elif biorhythms['intellectual_cycle'] < -0.5:
            advice_parts.append("Mental: Slower thinking period - avoid major decisions")
        
        return '\n'.join(advice_parts)

class BioTrackerSystem:
    """Main system orchestrator"""
    
    def __init__(self, person: Person):
        self.person = person
        self.cycle_calculator = BioCycleCalculator(person)
        self.hormone_calculator = HormoneCalculator(person)
        self.advice_generator = AdviceGenerator()
    
    def generate_daily_report(self, date: datetime.date = None) -> Dict:
        """Generate comprehensive daily report"""
        if date is None:
            date = datetime.date.today()
        
        biorhythms = self.cycle_calculator.calculate_biorhythms(date)
        hormones = self.hormone_calculator.calculate_seasonal_hormones(date)
        advice = self.advice_generator.generate_advice(biorhythms)
        
        return {
            'person': self.person.name,
            'date': date,
            'biorhythms': biorhythms,
            'hormones': hormones,
            'advice': advice
        }
    
    def print_report(self, date: datetime.date = None):
        """Print formatted report"""
        report = self.generate_daily_report(date)
        
        print(f"\n=== Bio Report for {report['person']} ===")
        print(f"Date: {report['date']}")
        print(f"Season: {report['biorhythms']['season'].value}")
        print(f"Moon Phase: {report['biorhythms']['moon_phase'].value}")
        print(f"Circadian Type: {report['biorhythms']['circadian_type'].value}")
        print(f"\nBiorhythm Cycles:")
        print(f"  Physical: {report['biorhythms']['physical_cycle']}")
        print(f"  Emotional: {report['biorhythms']['emotional_cycle']}")
        print(f"  Intellectual: {report['biorhythms']['intellectual_cycle']}")
        print(f"\nHormone Levels:")
        print(f"  Leptin: {report['hormones'].leptin}")
        print(f"  Ghrelin: {report['hormones'].ghrelin}")
        print(f"  Cortisol: {report['hormones'].cortisol}")
        print(f"\nPersonalized Advice:")
        print(report['advice'])

# Example usage
if __name__ == "__main__":
    # Create a person
    person = Person(
        name="John Doe",
        birth_date=datetime.date(1990, 5, 15),
        birth_time=datetime.time(8, 30),
        location=(40.7128, -74.0060),  # New York City
        bmi=25.0
    )
    
    # Create bio tracker system
    tracker = BioTrackerSystem(person)
    
    # Generate and print today's report
    tracker.print_report()
    
    # Generate report for specific date
    specific_date = datetime.date(2024, 6, 21)  # Summer solstice
    tracker.print_report(specific_date)
