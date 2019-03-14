"""
* Author : Batyr Nuryyev
* Date   : Feb 3, 2019
"""

import gym
import numpy as np
import operator
import copy
from neural_agent import NeuralAgent

class Evolution:
    def __init__(self, N = 10, K = 6):
        """
            Evolutionary NN optimization (finding best weights)
            
            env - cart pole environment (for reusability)
            N - size of the population (per each generation)
            best_K - selects best K performing individuals for crossover and next gen
            agents - list of neural networks (i.e. agents)
        """
        self.env = gym.make('CartPole-v0')
        self.N = N
        self.best_K = K
        self.agents = [NeuralAgent() for _ in range(self.N)]
        self.stats = []
        
    def reset(self):
        for agent in self.agents:
            agent.reset()
        
        #self.stats = []
    
    def crossover(self, ind1, ind2):
        """
            Single point crossover
        """
        # Are you ready, kids?!
        ind3 = NeuralAgent()
        ind4 = NeuralAgent()
        
        # Child 1: Aye-aye, Captain!
        ind3.w1 = ind2.w1 # input layer
        ind3.b1 = ind2.b1
        ind3.w2 = ind1.w2 # layer 2
        ind3.b2 = ind1.b2 
        ind3.w3 = ind2.w3 # layer 3
        ind3.b3 = ind2.b3

        # Child 2: Aye-aye, Captain!
        ind4.w1 = ind1.w1 # input layer
        ind4.b1 = ind1.b1
        ind4.w2 = ind2.w2 # layer 2
        ind4.b2 = ind2.b2
        ind4.w3 = ind1.w3 # layer 3
        ind4.b3 = ind1.b3
        
        # New offspring
        return (ind3, ind4)
    
    def select(self):
        # "Half & Half" crossover procedure
        scores = [ (agent, agent.reward) for agent in self.agents ]
        scores.sort(key = operator.itemgetter(1))
        scores.reverse()
        scores = scores[0 : len(scores) // 2]
        new_agents = [ agent for (agent, _) in scores ]
        
        # For stats (best performing agent's score is kept)
        self.stats.append(scores[0][1])
        
        # Crossover
        for i in range(len(scores) // 2):
            if len(new_agents) >= self.N:
                break
            
            # Choose two parents randomly
            f = np.random.randint(0, (len(scores) // 2) - 1)
            m = np.random.randint(0, (len(scores) // 2) - 1)
            
            if f == m:
                m = (m + 1) % (len(scores) // 2)
            
            # Sexy
            offspring = self.crossover(scores[f][0], scores[m][0])
            
            # Happy birthday, welcome to Jupyter notebook
            new_agents.append(offspring[0])
            new_agents.append(offspring[1])
        
        # Cut the tail! (i.e. if population size goes beyond N)
        if len(new_agents) > self.N:
            new_agents = new_agents[0 : self.N]
    
        self.agents = new_agents

    def mutate(self):
        p = np.random.random()
        
        if p < 0.10:
            # Mutate A random individuals
            A = 2

            for i in range(A):
                random_individual = np.random.choice(self.agents)
                random_individual.mutate()
    
    def evolve_step(self, n = 300, verbose = False):
        agent_c = 0
        for agent in self.agents:
            
            observation = self.env.reset()
            
            for t in range(n):
                observation, reward, done, _ = self.env.step(agent.action(observation))
                agent.add_reward(reward)
                
                if done:
                    break
            
            if verbose:
                print("Agent {0} reward = {1}".format(agent_c, agent.reward))
                
            agent_c += 1
    
    def getStats(self):
        # X is generation number
        X = [i + 1 for i in range( len(self.stats) )]
        Y = copy.deepcopy(self.stats)
        
        return (X, Y)

    def get_average(self):
        # Gets the mode of a list (most frequently appearing number)
        agents_scores = [agent.get_reward() for agent in self.agents]
        
        average = sum(agents_scores) / len(agents_scores)
        print("Average = {0}".format(average))
        
        return average
    
    def evolve(self, n):
        self.reset()
        verbose = False
        for i in range(n):
            
            if i % 50 == 0:
                print("Generation {0}".format(i))
                verbose = True
            else:
                verbose = False
            
            # Evolve!
            self.evolve_step(verbose = verbose)

            # Termination condition: if the average score is more than 199.
            # Checks only every 50 generations (it's expensive!)
            if i % 100 == 0 and self.get_average() > 198.0:
                # enough is enough!
                break

            self.select()
            self.mutate()
            
            # NOTE: I reset the score since accumulation of score of some genome
            # ruins the evolution. How? If a genome initially gets some score, even
            # though it performs not as good as initially, its score will get accumulated
            # and its weights gonna spread out to children (epidemic)
            # This is what fixed my convergence (getting a well-trained agent).
            self.reset()
    
    def get_best_agent(self):
        agents = copy.deepcopy(self.agents)
        agents.sort(key = lambda x : x.reward)        
        return agents[-1]
