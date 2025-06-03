def expanded_mmr(difficulty, context, processing_time, extra_energy):
    """
    Calculates the Manual Memory Recall (MMR) using the expanded equation.

    Args:
        difficulty: The difficulty of the recall task (0.0-1.0, higher = more difficult).
        context: The context in which the information was stored (0.0-1.0, higher = better context).
        processing_time: The time it takes to retrieve the information (seconds, positive float).
        extra_energy: The additional energy required for manual recall (positive float).

    Returns:
        The Manual Memory Recall (MMR) score (float).
        
    Raises:
        ValueError: If any parameter is outside valid range or negative.
    """
    
    # Input validation
    if not (0.0 <= difficulty <= 1.0):
        raise ValueError("Difficulty must be between 0.0 and 1.0")
    if not (0.0 < context <= 1.0):
        raise ValueError("Context must be between 0.0 (exclusive) and 1.0")
    if processing_time <= 0:
        raise ValueError("Processing time must be positive")
    if extra_energy <= 0:
        raise ValueError("Extra energy must be positive")

    # Incorporate difficulty into the calculation
    difficulty_factor = 1 + difficulty  # Scale difficulty impact
    
    # Calculate the numerator with difficulty factor
    numerator = (context * extra_energy * processing_time * difficulty_factor + 
                context * processing_time * processing_time + 
                extra_energy * processing_time * difficulty_factor)

    # Calculate the denominator
    denominator = context

    # Calculate the expanded Manual Memory Recall score
    mmr_score = numerator / denominator

    return mmr_score


def calculate_mmr_batch(recall_tasks):
    """
    Calculate MMR for multiple tasks.
    
    Args:
        recall_tasks: List of tuples (difficulty, context, processing_time, extra_energy)
        
    Returns:
        List of MMR scores
    """
    return [expanded_mmr(*task) for task in recall_tasks]


# Example usage
if __name__ == "__main__":
    # Single calculation
    difficulty = 0.7
    context = 0.5
    processing_time = 2.0
    extra_energy = 1.5

    try:
        mmr_score = expanded_mmr(difficulty, context, processing_time, extra_energy)
        print(f"Expanded Manual Memory Recall score: {mmr_score:.2f}")
        
        # Batch calculation example
        tasks = [
            (0.3, 0.8, 1.5, 1.0),  # Easy task
            (0.7, 0.5, 2.0, 1.5),  # Medium task
            (0.9, 0.3, 3.0, 2.0),  # Hard task
        ]
        
        batch_scores = calculate_mmr_batch(tasks)
        print("\nBatch MMR scores:")
        for i, score in enumerate(batch_scores, 1):
            print(f"Task {i}: {score:.2f}")
            
    except ValueError as e:
        print(f"Error: {e}")
