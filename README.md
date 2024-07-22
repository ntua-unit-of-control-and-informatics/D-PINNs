# Distributional Physics-Informed Neural Networks (D-PINNs)

D-PINN is a novel algorithm designed to enable statistical modeling within Physics-Informed Neural Networks (PINNs). Unlike traditional PINNs, which often focus on point estimates, D-PINNs integrate distributional assumptions directly into the optimization process. This integration allows D-PINNs to handle both forward and inverse problems by estimating state variables and system parameters via probability distributions, thereby effectively propagating uncertainty and interindividual variability observed in state variables through to parameter estimation and vice versa. The framework utilizes neural networks to predict statistical measures, such as the mean and variance, of state variables over time, and these predictions are then sampled to derive distributions for the variables and parameters. To demonstrate its capability to solve inverse and forward problems, the algorithm was applied to a pharmacokinetic modeling problem using a three-compartment model to simulate drug mass transfer in the body. Benchmarking of D-PINNs against a state-of-the-art algorithm, Hamiltonian Monte Carlo (HMC), highlighted the prediction accuracy and speed of convergence of the proposed methodology. The results suggest that D-PINNs offer a robust and efficient approach for incorporating statistical modeling into PINNs, significantly improving their applicability in fields such as pharmacokinetics, where incorporation of distributions to describe the population is crucial.

This repository provides the code that implements D-PINNS for both forward and inverse problems on an application of a simple 3-compartmental pharmacokinetic model. For more information about the theoretical background and the application of D-PINNs please refer to **D-PINNS: A simple algorithm for facilitating statistical modelling in PINNs**.

![D-PINNs Diagram](D-PINNs_schematic.png)