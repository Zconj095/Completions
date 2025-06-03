# main.py

from color_input_handler import get_aura_color
from personality_traits_mapper import get_personality_traits
from trait_display import display_traits
from aura_color_validation import is_valid_color

if __name__ == "__main__":
    aura_color = get_aura_color()
    if is_valid_color(aura_color):
        personality_traits = get_personality_traits(aura_color)
        display_traits(aura_color, personality_traits)
    else:
        print("Entered an invalid aura color. Please try again with a known color.")
