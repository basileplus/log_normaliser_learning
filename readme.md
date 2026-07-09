This repository contains the code to train and evaluate a neural network to learn the $\theta_\psi(\eta)$, mapping between natural parameter $\theta$ and its dual $\eta$ in exponential families.

We get sample from an unknown family of probability distributions via the `Sampler.py`file. Those samples are used to manually computed an unbiased estimate of the gradient of the following loss function

$$
\mathcal L(\psi) = A(\theta_\psi(\eta)) + A^\star(\eta) - \langle \theta_\psi(\eta), \eta \rangle
$$

#Files

- `Sampler.py` : class used to instanciate a sampler from which we retrieve samples for a given parameter and can compute the sufficient statistic
- `*_experiment.py`: each file correspond to a different experiment, namely a different family of distributions target. Those files contain the main SGD algorithm and the manual computation of the gradient
- `ICNN.py`: define ICNN used to parametrize our mapping. We learn $A^\star_\psi(\eta)$ via an ICNN and deduce $\theta_\psi(\eta)=\nabla A^\star_\psi(\eta)$
- `Visualizer.py` : contain some code to export figures of the experiment
- `experiment_logger.py`: creates a file to save into a JSON parameters of every runned experiment
