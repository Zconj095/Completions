import cupy as cp
import neat
from .neat_utils import create_neat_config
from .data_generation import generate_hexagonal_data


def eval_genomes(genomes, config):
    X, y = generate_hexagonal_data(10, 5, 2)
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        output = net.activate(X[0].ravel())
        genome.fitness = float(-cp.linalg.norm(output - y[0]))


def run_evolution():
    config = create_neat_config()
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    winner = pop.run(eval_genomes, n=5)
    return winner
