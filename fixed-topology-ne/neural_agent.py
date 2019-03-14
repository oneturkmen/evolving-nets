"""
* Author : Batyr Nuryyev
* Date   : Feb 3, 2019
"""

import numpy as np

class NeuralAgent:
    def __init__(self):
        """ The network with 4 layers in total:
            1) 4 input neurons
            2) 2 hidden neurons
            3) 1 hidden neuron
            4) 1 output neuron

            w1 - weights for the 1st hidden layer (2 neurons)
            b1 - bias for the 1st hidden layer (2 neurons)
            w2 - weights for the 2nd hidden layer (1 neuron)
            b2 - bias for the 2nd hidden layer (1 neuron)
            reward - agent's (network) reward/fitness
        """
        self.w1 = np.random.normal(0, 1, 8).reshape((2,4))
        self.b1 = np.random.normal(0, 1, 2).reshape((2,1))
        
        self.w2 = np.random.normal(0, 1, 4).reshape((2,2))
        self.b2 = np.random.normal(0, 1, 2).reshape((2,1))
        
        self.w3 = np.random.normal(0, 1, 2).reshape((1,2))
        self.b3 = np.random.normal(0, 1, 1)
        
        self.reward = 0
    

    def relu(self, z):
        """
            Rectified Linear Unit (ReLU)
        """
        return np.maximum(0, z)


    def tanh(self, z):
        """
            Hyperbolic tangent
        """
        return np.tanh(z)


    def sigmoid(self, z):
        """
            Good ol' sigmoid function
        """
        return 1 / (1 + np.exp(-z))
    
    
    def reset(self):
        """
            Reset the reward for a new episode
        """
        self.reward = 0
    
    
    def add_reward(self, r):
        """
            Adds reward per episode
        """
        self.reward += r
    
    def get_reward(self):
        """
            Returns the reward
        """
        return self.reward
    
    def mutate(self):
        """
            Mutation with no extreme abberation
        """
        
        # Mutate each weight
        self.w1 = self.w1 + np.random.normal(0, 1, 8).reshape((2,4))
        self.b1 = self.b1 + np.random.normal(0, 1, 2).reshape((2,1))
        self.w2 = self.w2 + np.random.normal(0, 1, 4).reshape((2,2))
        self.b2 = self.b2 + np.random.normal(0, 1, 2).reshape((2,1))
        self.w3 = self.w3 + np.random.normal(0, 1, 2).reshape((1,2))
        self.b3 = self.b3 + np.random.normal(0, 1, 1)
        
        # Return thyself
        return self
    
    
    def action(self, observations):
        """
        Simple forward propagation.
        
        observations - environment's observation output 
                       which will be fed into the neural network.
        
        returns - action (left or right movement)
        """
        # First hidden layer
        z1 = np.dot(self.w1, observations).reshape((2,1)) + self.b1
        a1 = self.relu(z1)
        
        # Second hidden layer
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.relu(z2)

        # Third layer (output
        z3 = np.dot(self.w3, a2) + self.b3
        a3 = self.tanh(z3)
        
        # Get the output 
        return 1 if a3 >= 0 else 0


nn = NeuralAgent()
action = nn.action([1,2,3,4])
print(action)