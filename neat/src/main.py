"""
* Author : Batyr Nuryyev
* Date   : April 23, 2019
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from evolution import Evolution
from visual import visualize

def run():
    # Initialize
    darwinian = Evolution(15, 6)

    # Evolve
    darwinian.evolve(150)

    # Get the best agent
    best_genome = darwinian.get_best_genome()
    
    # Visualize the performance
    for i_episode in range(10):
        observation = darwinian.env.reset()
        for t in range(500):
            darwinian.env.render()
            
            action = best_genome.action(observation)
            observation, reward, done, info = darwinian.env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    
    # Display the network architecture
    visualize(best_genome.get_inputs(), 
        best_genome.get_outputs(), best_genome.get_connections())
    
    # Plot the statistics
    stats = darwinian.get_statistics()
    plt.plot(stats[0], stats[1]) # average
    plt.plot(stats[0], stats[2]) # best
    plt.ylabel("Reward")
    plt.xlabel("Generation")
    plt.legend(['average', 'best'], loc='upper left')
    plt.show()
    
    # Sleep a bit
    time.sleep(1)
    darwinian.env.close()


if __name__ == "__main__":
    run()