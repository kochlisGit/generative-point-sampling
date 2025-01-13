# GPS: A Generative Point Sampling Apporach for PINNs (Journal)

This repository contains the implementation code of the work: "GPS: A Generative Point Sampling Apporach for PINNs (Journal)"

# Physics-Informed Neural Networks (PINNs)

Unlike traditional Neural Networks that learn to map their inputs to the corresponding targets, defined as $x \in \mathbb{R}^D$ and $y \in \mathbb{R}$ respectively, PINNs integrate physical laws, described by PDEs, directly into the learning procedure. This is achieved by adding the known differential equations, Boundary Conditions (BCs) and Initial Conditions (ICs), directly into the loss function, when training the network \citep{lagaris1998artificial}. This is accomplished by sampling $N_{train}$ points from a defined domain and feeding them into the network. Then, automatic differentiation is employed to calculate the gradients of the network's outputs with respect to the input points \citep{lagaris1998artificial}. Therefore, the residual of the underlying differential equation is computed using these gradients, defined as $L_{PDE}(\theta)$. Finally, the loss terms of BCs and ICs are computed, defined as $L_{BC}(\theta)$ and $L_{IC}(\theta)$ respectively, since their desired outputs, defined as $Y_{BC}$ and $Y_{IC}$ are known. The overall loss function can then be formulated as the sum of the three loss components, as presented in the following Equation:

$L(\theta) = L_{PDE}(\theta) + L_{BCs}(\theta) + L_{ICs}(\theta)$

where:
- $L_{PDE}(\theta)$ represents the loss from the partial differential equation (PDE)
- $L_{BCs}(\theta)$ represents the loss from the boundary conditions (BCs)
- $L_{ICs}(\theta)$ represents the loss from the initial conditions (ICs)

During training, the Neural Network tries to minimize the loss function error, using standard optimization algorithms, such as Stochastic Gradient Descent (SGD) or ADAM, and doing so, it learns the solution of the defined PDE, e.g. Burgers Equation, as presented in the Figure below:

![PINN Architecture](https://github.com/kochlisGit/generative-point-sampling/blob/main/figs/pinn_arch.png)
