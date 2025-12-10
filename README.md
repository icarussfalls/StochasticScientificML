# ScientificML

This repo contains the work I have done and followed in the Scientific ML course from ETH Zurich.

Since, I am currently studying MSc Financial Mathematics and my interest lies towards scienfific ml and pdes, I thought why not combine both pdes and stochastic in one and make a stochastic variant of the pdes. This is more realistic than the standard pdes.

In the StochasticHeat1D, I applied stochastic term to the second spatial derivative and added to the pde loss. The PINNs approximates the solution to the utmost accuracy, and comparing with the actual solution, the L2 relative error was 1%.

My target now is to go through inverse PDE with the stochastic variants and them implement FNO gradualy.