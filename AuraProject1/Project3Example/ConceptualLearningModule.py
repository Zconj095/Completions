import neat
import os
import json
import glob

def load_latest_analysis_results():
    """Load the latest analysis result for learning."""
    list_of_files = glob.glob('analysis_result_*.json')
    latest_file = max(list_of_files, key=lambda x: x.split('_')[2])
    with open(latest_file, 'r') as file:
        analysis_result = json.load(file)
    return analysis_result

def evaluate_genome(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        analysis_result = load_latest_analysis_results()
        
        # This is a placeholder for your evaluation logic
        # For demonstration, let's assume a binary health status and convert it to a fitness score
        if analysis_result['health_status'] == "Healthy":
            genome.fitness = 1.0
        else:
            genome.fitness = 0.0

def run_neat_evolution():
    """Run the NEAT algorithm to evolve the analysis based on latest results."""
    config_path = 'ConceptualLearningModule-neat-config.txt'  # Make sure this path points to your NEAT config file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))

    winner = p.run(evaluate_genome, 10)  # Run for a limited number of generations for demonstration

    # Save or apply the winning genome's logic to your analysis model
    print("Evolved a new analysis model.")

if __name__ == "__main__":
    run_neat_evolution()
