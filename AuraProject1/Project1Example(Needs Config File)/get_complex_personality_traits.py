def get_complex_personality_traits(aura_color):
    # Handling direct matches and simple categories first
    if aura_color.lower() in traits_map:
        return traits_map[aura_color.lower()]

    # Decomposing the input for mixed and transition colors
    traits = []
    for color in aura_color.split(","):
        color = color.strip().lower()
        if color in traits_map:
            traits.append(traits_map[color])
        else:
            # Check for specific transitions or complex color definitions
            for transition, description in complex_transitions_map.items():
                if transition in aura_color.lower():
                    traits.append(description)
                    break
            else:
                traits.append("Unknown aura color. Unable to determine personality traits.")

    # Combining traits for mixed colors
    if "mixed colors" in aura_color.lower():
        traits.append(traits_map["mixed colors"])

    return ". ".join(set(traits))

def display_complex_traits(aura_color):
    personality_traits = get_complex_personality_traits(aura_color)
    print(f"Aura Color(s): {aura_color.capitalize()}\nPersonality Traits: {personality_traits}")