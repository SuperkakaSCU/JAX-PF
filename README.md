__JAX-PF__: an efficient GPU-computing simulation for differentiable phase field (PF) simulaiton, built on top of [JAX-FEM](https://github.com/deepmodeling/jax-fem) leveraging [JAX](https://github.com/google/jax). 

## The concept of differentiable PF
We want to emphasize the following four features that differential JAX-PF from other PF software:
- __Ease-of-use__: Leveraging capability of automatic differentiation (AD) in JAX, JAX-PF automatically generates Jacobians and derivatives of different energy terms with machine precision, enabling the realization of multi-physics and multi-variable PF models.
- __Automatic Sensitivity__: Implicit time integration with customized adjoint-based AD enables efficient gradient-based optimization and inverse design of strongly nonlinear PF systems.
- __High-performance GPU-acceleration__: Through the XLA backend and vectorized operations, JAX-PF delivers competitive GPU performance, drastically reducing computational time relative to CPU-based solvers.
- __Unified multiscale ecosystem with JAX-CPFEM__: Built on the same JAX-FEM foundation, JAX-PF integrates seamlessly with [JAX-CPFEM](https://github.com/SuperkakaSCU/JAX-CPFEM) to enable coupled process–structure–property simulations (e.g., dynamic recrystallization), while preserving full differentiability for optimization and design.
  

:fire: ***Join us for the development of JAX-PF! This project is under active development!***

<br>

## Applications
### Benchmarks
Four benchmark problems are provided, including [Allen–Cahn](https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation), [Cahn–Hilliard](https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation), coupled Allen–Cahn and Cahn–Hilliard, each implemented with both explicit and implicit time integration, and [Eshelby inclusion](https://en.wikipedia.org/wiki/Eshelby%27s_inclusion) for lattice misfit in solid-state phase transformations.

:mega: Comparison between JAX-PF and [PRISMS-PF](https://github.com/prisms-center/phaseField).


<p align="middle">
  <img src="docs/materials/benchmarks/Validation_benchmark.jpg" width="800" />
<p align="middle">
    <em >Validation of benchmark problems in JAX-PF. </em>
</p>


<br>
<br>

### Forward Case Studies
 
:fire: ***For each case, both explicit and implicit time stepping schemes are provided***

<p align="middle">
  <img src="docs/materials/applications/Case1_IC.png" width="220" />
  <img src="docs/materials/applications/Case1_F.png" width="220" />
<p align="middle">
    <em >The initial (left) and final (right) grain structure for a 2D grain growth simulation.</em>
</p>
<br>
<br>


<p align="middle">
  <img src="docs/materials/applications/Case2_IC.png" width="300" />
  <img src="docs/materials/applications/Case2_F.png" width="300" />
<p align="middle">
    <em >The distribution of composition during a simulation of spinodal decomposition from initial fluctuations (left) to final two distinct phases (right).</em>
</p>
<br>
<br>


<p align="middle">
  <img src="docs/materials/applications/Case3.png" width="550" />
<p align="middle">
    <em >A 2D simulations of the multi-variants precipitate in an Mg-Nd alloy.</em>
</p>
<br>
<br>


<p align="middle">
  <img src="docs/materials/applications/Case4.png" width="550" />
<p align="middle">
    <em >A 3D simulations of the single-variants precipitate in an Mg-Nd alloy.</em>
</p>


<div align="center">
  <img src="docs/materials/applications/Case5.gif" width="550" />
  <div><em>Static recrystallized microstructure.</em></div>
</div>


<br>
<br>

### Multiscale Simulations
:mega: Multiscale simulations (PF-CPFEM) using JAX-PF and [JAX-CPFEM](https://github.com/SuperkakaSCU/JAX-CPFEM), which are built on top of the same underlying [JAX-FEM](https://github.com/deepmodeling/jax-fem) ecosystem.


<p align="middle">
  <img src="docs/materials/applications/CPPF.png" width="800" />
<!-- <p align="middle">
    <em >Coupled JAX-PF and JAX-CPFEM framework for process–structure–property integration.</em>
</p> -->



### Inverse Design
:mega: A demos: calibration of material parameters.

<p align="middle">
  <img src="docs/materials/inverse/calibration.png" width="550" />
<!-- <p align="middle">
    <em >Calibration of material parameters using differentiable PF simulations for multi-variants in Mg-Nd alloys. Subfigure (a) shows the evolution of the objective function with the number of optimization iterations, based on the synthetic imaging of the microstructure as the ground truth, as shown in the bottom subfigure in Fig. 6. Subfigure (b) shows the comparison between reference and calibrated material parameters.Subfigure (c) shows the comparison between the reference (red) and calibrated (blue) microstructural morphology of the different variants, with precipitate variants identified by concentrations in the range 0.12 to 0.16. Subfigure (d) and (e) shows the Concentration distributions for the reference and calibrated results across the domain.</em>
</p> -->
<br>
<br>



## Installation
JAX-PF supports Linux and macOS, which depend on JAX-FEM.
### Install JAX-FEM
JAX-FEM is a collection of several numerical tools, including the Finite Element Method (FEM). See JAX-FEM installation [instructions](https://github.com/deepmodeling/jax-fem?tab=readme-ov-file). Depending on your hardware, you may install the CPU or GPU version of JAX. Both will work, while the GPU version usually gives better performance.

### Install Neper
[Neper](https://neper.info/) is a free/open-source software package for polycrystal generation and meshing. It can be used to generate polycrystals with a wide variety of morphological properties. A good [instruction](https://www.youtube.com/watch?v=Wy9n756wFu4&list=PLct8iNZXls-BMU7aleWoSoxgD8OFFe48W&index=5) video is on Youtube.




### Install JAX-PF
Place the downloaded `phaseField/` file in the `applications/` folder of JAX-FEM, and then you can run it.

### Quick Tests
For example, you can download `phaseField/allenCahn/explicit_fem` folder and place it in the `applications/` folder of JAX-FEM, run
```bash
python -m applications.phaseField.allenCahn.explicit_fem.explicit_AC
```
from the root directory. Use [Paraview](https://www.paraview.org/) for visualization.


## Tutorial
We shared the introduction of JAX-PF on [arXiv](https://arxiv.org/abs/2601.06079), which contains an inverse microstructure design case.

## Citations
If you found this library useful in academic or industry work, we appreciate your support if you consider 1) starring the project on Github, and 2) citing relevant papers:
1) Efficient GPU-computing simulation platform JAX-PF for differentiable phase field model.
   DOI: https://arxiv.org/abs/2601.06079
2) Efficient GPU-computing simulation platform JAX-CPFEM for differentiable crystal plasticity finite element method.
   DOI: https://doi.org/10.1038/s41524-025-01528-2
