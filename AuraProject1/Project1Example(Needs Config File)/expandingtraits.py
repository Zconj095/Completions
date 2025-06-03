import json
def load_traits():
    with open("traits.json", "r") as file:
        return json.load(file)

def save_traits(traits):
    with open("traits.json", "w") as file:
        json.dump(traits, file, indent=4)

def add_trait(color, new_trait):
    traits = load_traits()
    if color in traits:
        if new_trait not in traits[color]:
            traits[color].append(new_trait)
            save_traits(traits)
            print(f"Added new trait '{new_trait}' to color '{color}'.")
        else:
            print(f"Trait '{new_trait}' already exists for color '{color}'.")
    else:
        print(f"Color '{color}' not found. Adding color with new trait.")
        traits[color] = [new_trait]
        save_traits(traits)

def get_complex_personality_traits(aura_color):
    traits = load_traits()
    # Your existing logic here for interpreting colors and returning traits
