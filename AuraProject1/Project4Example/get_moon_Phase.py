import datetime
import ephem
import random
def get_moon_phase(date):
    observer = ephem.Observer()
    observer.date = date.strftime("%Y/%m/%d")
    moon = ephem.Moon(observer)
    moon_phase_number = moon.phase / 100

    if moon_phase_number == 0:
        return "New Moon"
    elif 0 < moon_phase_number <= 0.25:
        return "Waxing Crescent"
    elif 0.25 < moon_phase_number <= 0.5:
        return "First Quarter"
    elif 0.5 < moon_phase_number <= 0.75:
        return "Waxing Gibbous"
    elif 0.75 < moon_phase_number < 1:
        return "Full Moon"
    elif 1 > moon_phase_number > 0.75:
        return "Waning Gibbous"
    elif 0.75 > moon_phase_number > 0.5:
        return "Last Quarter"
    elif 0.5 > moon_phase_number > 0:
        return "Waning Crescent"
    
def calculate_biorhythms(date):
    moon_phase = get_moon_phase(date)
    sun_cycle_phase = get_sun_cycle_approx(date)
    
    print(f'Date: {date}')
    print(f'Moon Phase: {moon_phase}')   
    print(f'Sun Cycle Phase: {sun_cycle_phase}') 
    

def get_season(date):
    # Placeholder for season calculation
    month = date.month
    if 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 8:
        return "Summer"
    elif 9 <= month <= 11:
        return "Autumn"
    else:
        return "Winter"

def get_circadian_tendency(birth_time):
    # Placeholder for circadian rhythm calculation
    if birth_time.hour < 12:
        return "Morning Person"
    else:
        return "Evening Person"

def calculate_biorhythms(date, birth_date, birth_time):
    season = get_season(date)
    moon_phase = get_moon_phase(date)
    sun_cycle_phase = get_sun_cycle_approx(date)
    circadian_tendency = get_circadian_tendency(birth_time)

    
    # Print out calculated cycles
    print(f'Date: {date}')
    print(f'Season: {season}')
    print(f'Moon Phase: {moon_phase}')   
    print(f'Sun Cycle Phase: {sun_cycle_phase}')
    print(f'Circadian Tendency: {circadian_tendency}') 
    
    
def get_sun_cycle_approx(current_date):
    """
    Approximate the solar cycle phase based on the current date.
    This is a simplified method and may not be highly accurate.
    """
    # Approximate length of the solar cycle in years
    solar_cycle_length = 11

    # A recent solar cycle began in 2020
    cycle_start_year = 2020

    # Calculate the current year in the cycle
    year_in_cycle = (current_date.year - cycle_start_year) % solar_cycle_length

    # Determine the sun cycle phase
    if year_in_cycle < 3:
        return "Rising Phase"
    elif 3 <= year_in_cycle < 5:
        return "Solar Maximum"
    elif 5 <= year_in_cycle < 8:
        return "Declining Phase"
    else:
        return "Solar Minimum"

# Test the function with the current date
# Example usage
test_date = datetime.datetime(2024, 2, 3)
print(get_moon_phase(test_date))
get_sun_cycle_approx(test_date)

    


# Example usage
birth_info = datetime.datetime(1995, 3, 6, 14, 0, 0)  
today = datetime.datetime(2024, 1, 20)
calculate_biorhythms(today, birth_info, birth_info.time())
