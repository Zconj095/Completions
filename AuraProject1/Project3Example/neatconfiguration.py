import os
import neat
import numpy as np

# Assuming 'neat-config.txt' is in the same directory as this script
config_path = os.path.join(os.path.dirname(__file__), 'neat-config.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Generate test data and expected outputs
def generate_test_data(num_samples=100, num_features=10):
    # Generate random values from -1 to 1
    inputs = np.random.rand(num_samples, num_features) * 2 - 1
    expected_outputs = np.sin(inputs)  # Simplified expected output calculation
    return inputs, expected_outputs

# Define the network evaluation function
def evaluate_net(net, inputs, expected_outputs):
    # Activate the network on each input
    error_sum = 0
    for input, expected in zip(inputs, expected_outputs):
        output = net.activate(input)
        # Here, assuming a single output and expected output for simplicity
        error = (output[0] - expected[0]) ** 2
        error_sum += error
    # Average error
    error_avg = error_sum / len(inputs)
    # Convert error to fitness (higher is better, so subtract from 1)
    fitness = 1 - error_avg
    return fitness

# Evaluation function for genomes
def evaluate_genome(genomes, config):
    inputs, expected_outputs = generate_test_data()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = evaluate_net(net, inputs, expected_outputs)
        genome.fitness = fitness

# Create NEAT population and add reporters
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))

# Run NEAT evolution using the evaluation function
winner = p.run(evaluate_genome, 50)  # Run for 50 generations

if __name__ == '__main__':
    print("\nBest genome:\n{!s}".format(winner))
