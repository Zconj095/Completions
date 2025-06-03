# personality_traits_mapper.py

def get_personality_traits(aura_color):
    # Expanded traits map with complex color categories
    traits_map = {
        "red": "Passionate, energetic, and competitive",
        "orange": "Creative, adventurous, and confident",
        "yellow": "Optimistic, cheerful, and intellectual",
        "green": "Balanced, natural, and stable",
        "blue": "Calm, trustworthy, and communicative",
        "indigo": "Intuitive, curious, and reflective",
        "violet": "Imaginative, visionary, and sensitive",
        "warm colors": "Vibrant, sociable, and dynamic",
        "cool colors": "Reflective, calm, and thoughtful",
        "neutral colors": "Sophisticated, professional, and grounded",
    }

    # Handling complex and mixed color inputs
    complex_colors_map = {
        "warm": "warm colors",
        "cool": "cool colors",
        "neutral": "neutral colors",
    }

    # Check for direct matches first
    if aura_color.lower() in traits_map:
        return traits_map[aura_color.lower()]

    # Check for complex color categories
    for complex_category, complex_trait in complex_colors_map.items():
        if complex_category in aura_color.lower():
            return traits_map[complex_trait]

    # Fallback for unhandled inputs
    return "Unknown aura color. Unable to determine personality traits."

def analyze_color_combinations(aura_color):
    # Split the input by common conjunctions or commas
    color_keywords = aura_color.lower().replace("and", ",").split(",")
    color_keywords = [keyword.strip() for keyword in color_keywords]

    # Collect traits for each recognized color component
    traits = []
    for color in color_keywords:
        trait = get_personality_traits(color)
        if "Unknown" not in trait:
            traits.append(trait)

    # Combine unique traits
    combined_traits = ". ".join(set(traits))

    return combined_traits if combined_traits else "Unable to determine personality traits from given colors."

# Extended traits map with complex color categories and transitions
traits_map = {
    "red": "Passionate, energetic, and competitive",
    "blue": "Calm, trustworthy, and communicative",
    "yellow": "Optimistic, cheerful, and intellectual",
    "orange": "Creative, adventurous, and confident",
    "warm colors": "Vibrant, sociable, and dynamic",
    "cool colors": "Reflective, calm, and thoughtful",
    "neutral colors": "Sophisticated, professional, and grounded",
    "gray": "Balanced, neutral, and flexible",
    "silver": "Futuristic, graceful, and elegant",
    "mixed colors": "Complex, multifaceted, and adaptable",
}

# New entries for complex transitions
complex_transitions_map = {
    "monochrome to gray": "Transitioning from stark contrast to balance",
    "gray to silver": "Evolving from neutrality to sophisticated vibrancy",
}
