# aura_color_validation.py
def is_valid_color(aura_color):
    valid_colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
    return aura_color.lower() in valid_colors