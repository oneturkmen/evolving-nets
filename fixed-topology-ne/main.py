"""
* Author : Batyr Nuryyev
* Date   : Feb 3, 2019
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from evolution import Evolution

def test():
    # Initialize
    darwinian = Evolution(10, 5)

    # Evolve
    darwinian.evolve(1000)

    # Get the best agent
    best_agent = darwinian.get_best_agent()
    
    # Visualize the performance
    for i_episode in range(10):
        observation = darwinian.env.reset()
        for t in range(500):
            darwinian.env.render()
            
            action = best_agent.action(observation)
            observation, reward, done, info = darwinian.env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    
    # Plot the statistics
    stats = darwinian.getStats()
    plt.plot(stats[0], stats[1])
    plt.ylabel("Best Rewarded Genome")
    plt.xlabel("Generation")
    plt.show()
    
    # Sleep a bit
    time.sleep(1)
    darwinian.env.close()


if __name__ == "__main__":
    test()
