"""
* Author : Batyr Nuryyev
* Date   : March 14, 2019
"""

"""
The following class will contain the population of genomes,
evaluation of each through fitness function (openAI gym in this case),

SELECTION: Reproduction happens in a separate class
MUTATION:  Each genome will mutate on its own. Function is called in Selection

* Population
* Variables:
    - genomes
    - generation
    - environment
    - constants c1/c2 for speciation?
"""

import gym
import numpy as np
import operator
import copy

from genome import Genome
from reproduction import crossover


class Evolution:
    """ Driver (core) class which performs the "evolution".
        Here, we gather everything we've implemented so far
        for NEAT.
    """

    def __init__(self, N = 10, K = 5, C1 = 1, C2 = 1, C3 = 1, env_dims = (4, 1)):
        """ Initializes the population, sets the environment,
            keeps track of the best genome.

            Params:
                N - size of the population (number of individuals)
                K - keeps k best genomes every generation
        """

        # Sanity checks
        assert N > K, "Invalid N, K dimensions configuration!"

        # Hardcoded
        self.env = gym.make('CartPole-v0')
        self.env_dims = env_dims

        # Softcoded or dynamically computed
        self.genomes = self.__initialize_population(N)
        self.best_genome = None
        self.N = N
        self.best_K = K
        self.C1 = C1    # accent on excess genes
        self.C2 = C2    # accent on disjoint genes
        self.C3 = C3    # accent on average weight differences

        # Distances
        #self.distances = {}

        # TODO: later
        # self.stats = []

    ############ - Private methods - ############

    def __fitness_normalizer(self, genomes_scored):
        """ Normalizes the fitness of all genomes for speciation. """
        assert len(genomes_scored) > 0, "Genomes list cannot empty!"
        
        # Container for normalized genomes
        genomes_normed = []

        for i in range(len(genomes_scored)):
            # Get the number of disjoints and excess
            num_disjoint = 0
            num_excess   = 0
            N = -1

            for j in range(len(genomes_scored)):
                if i == j:
                    continue
                
                c_disjoint, c_excess, c_N = genomes_scored[i][0].distance(genomes_scored[j][0])

                num_disjoint += c_disjoint
                num_excess += c_excess
                N = max(N, c_N)

            # Score normalization
            score_normalized = 0
            norm_condition = self.C1 * num_disjoint + self.C2 * num_excess
            if norm_condition > 10:
                score_normalized = genomes_scored[i][1] / 10
            elif norm_condition > 5:
                score_normalized = genomes_scored[i][1] / 5
            elif norm_condition > 2:
                score_normalized = genomes_scored[i][1] / 2
            else:
                score_normalized = genomes_scored[i][1]

            genomes_normed.append((genomes_scored[i][0], score_normalized))
            
        return genomes_normed


    def __initialize_population(self, N):
        """ Initializes a population of genomes of size N. """
        assert N > 0, "N cannot be less than 1!"
        
        # Populate
        initial_population = [Genome() for i in range(N)]

        # Initialize the genomes
        for genome in initial_population:
            genome.initialize(*self.env_dims)        

        return initial_population
        
    def evolve_step(self, n = 300, verbose = False):
        """ Runs a Gym simulation n times, with actions and rewards. """
        genome_c = 0
        for genome in self.genomes:
            
            observation = self.env.reset()
            
            for t in range(n):
                observation, reward, done, _ = self.env.step(genome.action(observation))                
                genome.add_score(reward)
                
                if done:
                    break
            
            if verbose:
                print("Agent {0} reward = {1}".format(genome_c, genome.get_score()))
                
            genome_c += 1
        return
    

    def reset(self):
        """ Resets the genomes score for better performance metric. """
        for genome in self.genomes:
            genome.reset_score()
        return


    def selection(self):
        """ Selection of the best individuals as well as crossover. """
        assert len(self.genomes) > 0, "No individuals for selection!"

        # Rank the genomes and select only half of them
        genomes_scores = [(g, g.get_score()) for g in self.genomes]

        # Normalize the score of each (fitness sharing)
        genomes_scores_norm = self.__fitness_normalizer(genomes_scores)

        # Sort and take the best half
        genomes_scores_norm.sort(key = operator.itemgetter(1), reverse=True)
        genomes_scores_norm = genomes_scores_norm[0 : len(genomes_scores_norm) // 2]
        new_genomes = [ genome for (genome, _) in genomes_scores_norm ]

        # Crossover
        for i in range(len(genomes_scores_norm)):
            if len(new_genomes) >= self.N:
                break
            
            # Choose two parents randomly
            f = np.random.randint(0, (len(genomes_scores_norm) // 2) - 1)
            m = np.random.randint(0, (len(genomes_scores_norm) // 2) - 1)
            
            if f == m:
                m = (m + 1) % (len(genomes_scores_norm) // 2)
            
            # Sexy
            offspring = crossover(genomes_scores_norm[f][0], genomes_scores_norm[m][0])            
            
            # Happy birthday, kid!
            new_genomes.append(offspring)
        
        # Cut the tail! (i.e. if population size goes beyond N)
        if len(new_genomes) > self.N:
            new_genomes = new_genomes[0 : self.N]
    
        self.genomes = new_genomes

    
    def mutation(self):
        """ Mutation operation for population variation and diversity. """
        assert len(self.genomes) > 1, "No individuals for mutation!"
        
        # NOTE: probabilities are defined inside the genome method.
        for genome in self.genomes:
            genome.mutate()

        return
    

    ############ - Public methods - ############
    def get_best_genome(self):
        """ Gets best performing agent at the time when this func is called. """
        genomes = copy.deepcopy(self.genomes)
        genomes.sort(key = lambda λ : λ.get_score())
        return genomes[-1]

    def get_average(self):
        """ Returns average score in a population. """
        # Gets the mode of a list (most frequently appearing number)
        genomes_scores = [genome.get_score() for genome in self.genomes]
        
        average = sum(genomes_scores) / len(genomes_scores)
        print("Average = {0}".format(average))
        
        return average

    def evolve(self, n):
        """ Runs evolutionary process for n steps (generations). """
        assert n > 0, "Cannot evolve 0 or less generations!"

        # Prepare the genomes
        self.reset()
        
        # Whether the performance details should be printed
        verbose = False

        for i in range(n):

            # Print information every 50th time            
            if i % 50 == 0:
                print("Generation {0}".format(i))
                verbose = True
            else:
                verbose = False
            
            # Evolve!
            self.evolve_step(verbose = verbose)

            # Termination condition that depends on 
            # the maximum performance of a Gym environment
            #if i % 100 == 0 and self.get_average() > 198.0:
                # enough is enough!
                #break

            self.selection()
            self.mutation()
            
            # Reset the score every 10 generations for better performance metric
            if i % 10 == 0:
                self.reset()
    
    # TODO:
    # def get_best_genome():
    #   pass
    # 
    # def get_statistics():
    #   pass
    #

testing = False
if testing:
    darwin = Evolution()
    #darwin.selection()

    genomes_scores = [(g, 10+np.random.normal())for g in darwin.genomes]
    print(genomes_scores)    
    print("\n\n")
    darwin.evolve(500)

    