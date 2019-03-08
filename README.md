## Evolving Neural Networks (WIP)

### About

There is a traditional way to optimize neural network's weights - through backpropagation and gradient descent. Though it is quite effective in terms of locating optima, it does not always end up in a global optima (which we may wish for certain, complex tasks). If we are given complex fitness landscape and we should its global optima, then the gradient descent method is likely to fail at this task. However, metaheuristics (including evolutionary algorithms) are very good in terms of finding a global optimum, especially for tasks with very variable, noisy fitness landscapes. 

Here, I will use Neuroevolution of Augmenting Topologies (NEAT), which optimizes neural network's weights as well as its topology [1]. I intend to implement it from scratch and test it with some OpenAI gym environment like balancing a cart pole or possibly MuJoCo (where "human" learns how to walk or stand up).

## What shall be done

- [x] Implement fixed-topology network and its evolution (e.g. for balancing a cart pole from OpenAI gym)
- [ ] Implement NEAT (with some OpenAI environment)
- [ ] Genotype to phenotype visualization (using Keras without evaluation)
- [ ] Testing at each of progress
- [ ] Incorporate novelty search for NEAT
- [ ] Check plasticity and whether it makes sense to incorporate it as well (i.e. for evolution)
- [ ] Co-evolutionatiry methods? Research needed.

## Contents

In this repo, you will find the implementation of NEAT and potentially its variations; an implementation of neuroevolution of a fixed topology (artificial) neural network is also present.

### References

[1] - Kenneth O. Stanley and Risto Miikkulainen. Evolving Neural Networks through Augmenting Topologies. *The MIT Press*, 2002.
