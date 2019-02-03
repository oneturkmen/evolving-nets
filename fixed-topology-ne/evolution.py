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
    def __init__(self):
        """
            Evolutionary NN optimization (finding best weights)
            
            env - cart pole environment (for reusability)
            N - size of the population (per each generation)
            best_K - selects best K performing individuals for crossover and next gen
            agents - list of neural networks (i.e. agents)
        """
        self.env = gym.make('CartPole-v0')
        self.N = 10
        self.best_K = 4
        self.agents = [NeuralAgent() for _ in range(self.N)]
        self.stats = []
        
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
        self.stats = []
    
    
    def crossover(self, ind1, ind2):
        """
            Single point crossover
        """
        ind3 = NeuralAgent()
        ind4 = NeuralAgent()
        
        ind3.w1 = ind1.w1
        ind3.b1 = ind1.b1
        ind3.w2 = ind2.w2
        ind3.b2 = ind2.b2
        
        ind4.w1 = ind2.w1
        ind4.b1 = ind2.b1
        ind4.w2 = ind1.w2
        ind4.b2 = ind1.b2
        
        # Return new offspring
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
        
        if p <= 0.10:
            # Mutate two random individuals
            for i in range(2):
                random_individual = np.random.choice(self.agents)
                random_individual.mutate()
    
    
    def evolve_step(self, n = 100, verbose = False):
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
    
    
    def evolve(self, n):
        self.reset()
        verbose = False
        for i in range(n):
            
            if i % 50 == 0:
                print("Generation {0}".format(i))
                verbose = True
            else:
                verbose = False
                
            self.evolve_step(verbose = verbose)
            self.select()
            self.mutate()
    
    def get_best_agent(self):
        agents = copy.deepcopy(self.agents)
        agents.sort(key = lambda x : x.reward)
        agents.reverse()
        return agents[0]
