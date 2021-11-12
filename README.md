# PDEandNN
## This project uses deep neural networks to solve and identify the Swift-Hohenberg equation; a fourth order stiff partial differential equation. This repository contains the data, code and technical report for this project.  This work is based off of the following papers:

- Maziar Raissi. ["Deep hidden physics models: Deep learning of nonlinear partial differential equations."](https://www.jmlr.org/papers/volume19/18-046/18-046.pdf) Journal of Machine Learning Research 19 (2018): 1–24.

- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. ["Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."](https://www.sciencedirect.com/science/article/pii/S0021999118307125) Journal of Computational Physics 378 (2019): 686-707.

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. ["Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations."](https://arxiv.org/abs/1711.10561) arXiv preprint arXiv:1711.10561 (2017).

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. ["Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations."](https://arxiv.org/abs/1711.10566) arXiv preprint arXiv:1711.10566 (2017).


## Overview 
[Project Description](#project-description)  
[Main Takeaways](#main-takeaways)  
[Generated Data](#generated-data)
[Results](#results)

## Project Description 
We review two deep learning models that concern the dynamics underlying sparse and/or noisy spatiotemporal data appearing in the above papers. The first model aims to infer the solution to a known partial differential equation from space and time data. The second model aims to identify the PDE itself. We apply these models to the 1-dimensional Swift-Hohenberg equation, which has not appeared in associated literature. See the [technical report](technical_report.pdf) for the relevant background, discussion and results. 

## Main Takeaways 
The first model was able to accurately solve the Swift-Hohenberg equation with unperturbed data [See SolvingSH1D-Class.ipynb](SolvingSH1D-Class.ipynb). Additionally the second model that seeks to learn the PDE was able to identify the parameters with reasonable accuracy under the presence of noise. However, the second model was not able to adequately learn the entire PDE from spatio-temporal data [See (IdentifyingSH1D-Class.ipynb)](IdentifyingSH1D-Class.ipynb).

For example, here are the true dynamics of the solution: 
![](pdfHD_rolls_1_true.pdf)

Here are the dynamics predicted by the model: 
![](pdfHD_rolls_1_pred_int.pdf)

The class file for the model that seeks to solve the PDE can be found [here](PhysicsInformedNN1.py). The class file for the model that seeks to learn the PDE itself can be found [here](DeepHiddenPhysicsModels1.py).

## Generated Data 
The data for this project was generated in Matlab using the Chebfun package. The file that generates the data can be found [here](sh_1d.m). The .mat files containing the data for two solutions to the Swift-Hohenberg equation can be found [here](exp1d_HD.mat) and [here](exp1d_HD_even.mat) 



