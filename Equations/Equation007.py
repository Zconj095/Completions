import math

def narrative_construction_function(vocabulary_richness, sequential_understanding, logical_coherence):
    """
    Calculates the narrative construction score using a weighted combination.
    
    Args:
        vocabulary_richness: Breadth and precision of language (0-1)
        sequential_understanding: Ability to order information (0-1)
        logical_coherence: Logical consistency maintenance (0-1)
    
    Returns:
        Weighted narrative construction score
    """
    # Weighted formula with emphasis on logical coherence
    weights = {'vocab': 0.3, 'sequence': 0.35, 'logic': 0.35}
    
    base_score = (
        weights['vocab'] * vocabulary_richness +
        weights['sequence'] * sequential_understanding +
        weights['logic'] * logical_coherence
    )
    
    # Apply non-linear enhancement for high-performing combinations
    synergy_bonus = math.sqrt(vocabulary_richness * sequential_understanding * logical_coherence)
    
    return base_score * (1 + 0.2 * synergy_bonus)

def micromanaged_memory(vocabulary_richness, sequential_understanding, logical_coherence, associative_fluency):
    """
    Calculates the Micromanaged Narrative (MN) score based on cognitive factors.

    Args:
        vocabulary_richness: Breadth and precision of language (0-1)
        sequential_understanding: Ability to accurately order information (0-1)
        logical_coherence: Ability to maintain logical consistency (0-1)
        associative_fluency: Ease of introducing relevant details (0-1)

    Returns:
        The calculated MN score with validation
    """
    # Input validation
    inputs = [vocabulary_richness, sequential_understanding, logical_coherence, associative_fluency]
    if not all(0 <= x <= 1 for x in inputs):
        raise ValueError("All input values must be between 0 and 1")

    narrative_construction_score = narrative_construction_function(
        vocabulary_richness, sequential_understanding, logical_coherence
    )

    mn_score = narrative_construction_score * associative_fluency
    
    return min(mn_score, 1.0)  # Cap at maximum score of 1.0

def analyze_cognitive_profile(mn_score, vocab, seq, logic, assoc):
    """Provides interpretative analysis of the cognitive profile."""
    print(f"\n--- Cognitive Analysis ---")
    print(f"MN Score: {mn_score:.3f}")
    print(f"Vocabulary: {vocab:.3f} | Sequencing: {seq:.3f}")
    print(f"Logic: {logic:.3f} | Associative: {assoc:.3f}")
    
    if mn_score > 0.8:
        print("Profile: High narrative capability")
    elif mn_score > 0.6:
        print("Profile: Moderate narrative capability")
    else:
        print("Profile: Developing narrative capability")

# Example usage with enhanced analysis
if __name__ == "__main__":
    vocabulary_score = 0.85
    sequencing_score = 0.72
    coherence_score = 0.91
    associative_score = 0.88

    try:
        mn_result = micromanaged_memory(vocabulary_score, sequencing_score, coherence_score, associative_score)
        print(f"Micromanaged Narrative score: {mn_result:.3f}")
        analyze_cognitive_profile(mn_result, vocabulary_score, sequencing_score, coherence_score, associative_score)
    except ValueError as e:
        print(f"Error: {e}")
