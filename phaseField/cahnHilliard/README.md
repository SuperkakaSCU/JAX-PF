# Cahn-Hilliard Equation

## 1. Introduction

The Cahn-Hilliard equation is a fundamental parabolic partial differential equation in phase-field modeling, used to describe phase separation in binary mixtures. Unlike the Allen-Cahn equation, it involves **conserved** dynamics and includes **fourth-order spatial derivatives**.

### General Variational Form

The Cahn-Hilliard equation is expressed as:

$$\frac{\partial c}{\partial t} = - \nabla \cdot \left(-\mathbf{M_{c}}\left(\nabla\left(\frac{\delta F_{tot}}{\delta c} \right)\right)\right)  \quad \text{in } \Omega$$

where:
- $c$ is the composition field (phase-field variable)
- $\mathbf{M_{c}}$ is the constant mobility (scalar in isotropic case)
- $F_{tot}$ is the total free energy


### Total Free Energy

The total free energy functional consists of two main contributions:

$$F_{tot}(c) = \int_\Omega \left[f(c) + \frac{\kappa}{2}|\nabla c|^2\right] d\Omega$$

where:

1. **Chemical Free Energy**: $\int_\Omega f(c) d\Omega$
   - Double-well (or multi-well) form favors phase separation
   - $f(c)$ is the local chemical free energy density
   - The derivative of chemical potential (driving force) is computed as: $$f'(c) = \frac{\partial f}{\partial c} = 4c(c - 1)(c - 0.5)$$

2. **Interfacial Energy**: $\int_\Omega \frac{\kappa}{2}|\nabla c|^2 d\Omega$
   - Arising from the gradient energy terms
   - $\kappa$ is the gradient energy coefficient


## 2. Operator Split Form (Mixed Formulation)

To avoid fourth-order derivatives in the weak form, the Cahn-Hilliard equation is reformulated as two coupled second-order equations:

$$\frac{\partial c}{\partial t} - \nabla \cdot \left( \mathbf{M_{c}}\nabla\mu \right)= 0 \quad \text{in } \Omega$$

$$\mu - \frac{\delta F_{tot}}{\delta c} = 0 \quad \text{in } \Omega$$

**Unknown fields:** $(c, \mu)$ where:
- $c$ is the concentration (conserved order parameter)
- $\mu$ is the chemical potential (auxiliary variable)
- $\frac{\delta F_{tot}}{\delta c} = f'(c) - \kappa \nabla^2 c$

This formulation allows the use of standard Lagrange finite element basis functions.


---

## 3. Time Discretization

### General $\theta$-Method Formulation

The θ-method provides a family of time integration schemes [1]. For the mixed form of Cahn-Hilliard:

$$\frac{c^{n+1} - c^n}{\Delta t} - \nabla \cdot \mathbf{M_{c}}\nabla\mu^{n+\theta} = 0$$

$$\mu^{n+1} - \frac{\partial f}{\partial c}(c^{n+1}) + \kappa \nabla^2 c^{n+1} = 0$$

where:

$$\mu^{n+\theta} = (1-\theta)\mu^n + \theta\mu^{n+1}$$

**Special Cases:**
- $\theta = 0$: Forward Euler
- $\theta = 0.5$: $\theta$-Method
- $\theta = 1$: Backward Euler

---

## 4. Weak Formulation (Implicit θ-Method)

### Mixed Weak Form

Find $(c^{n+1}, \mu^{n+1}) \in V \times V$ such that for all test functions $(q, w) \in V \times V$:

$$\int_\Omega \frac{c^{n+1} - c^n}{\Delta t} q d\Omega + \int_\Omega \mathbf{M_{c}}\nabla\mu^{n+\theta} \cdot \nabla q d\Omega = 0$$

$$\int_\Omega \mu^{n+1} w d\Omega - \int_\Omega f'(c^{n+1}) w d\Omega - \int_\Omega \kappa \nabla c^{n+1} \cdot \nabla w d\Omega = 0$$

### Implementation Details

Implementation:  [implicit_fem/implicit_CH_theta.py](implicit_fem/implicit_CH_theta.py)
The weak form is assembled from six main contributions:

#### Equation 1 (Phase Field Concentration):

1. **Time Evolution Term:**
   ```
   val1 = ∫ (cⁿ⁺¹ - cⁿ)/Δt · q dΩ
   tmp1 = (c - c_old) / dt
   ```

2. **Diffusion Term (via chemical potential):**
   ```
   val2 = ∫ M_c·∇μⁿ⁺ᶿ·∇q dΩ
   tmp2 = M_c * ∇μⁿ⁺ᶿ
   ∇μⁿ⁺ᶿ = (1 - θ)·μⁿ + θ·μⁿ⁺¹
   ```

**Total for Equation 1:**
$$\text{RES}_c = \text{val1} + \text{val2} = 0$$

#### Equation 2 (Chemical Potential):

3. **Chemical Potential (LHS):**
   ```
   val3 = ∫ μⁿ⁺¹·w dΩ
   tmp3 = μ
   ```

4. **Free Energy Derivative (RHS):**
   ```
   val4 = -∫ f'(cⁿ)·w dΩ
   tmp4 = vmap_dfdc_func(p_old)  # f'(c) evaluated at old time step (or at current time step using p)
   ```

5. **Gradient Energy Term (RHS):**
   ```
   val5 = -∫ κ·∇cⁿ⁺¹·∇w dΩ
   tmp5 = KnV * p_grads  # κ times gradient of current c
   ```

**Total for Equation 2:**
$$\text{RES}_\mu = \text{val3} + \text{val4} + \text{val5} = 0$$


---

## 5. Implementation Schemes

### Directory Structure

```
cahnHilliard/
├── implicit_fem/          # Implicit θ-method solver
│   ├── implicit_CH_theta.py     # Main implicit solver with θ-method
│   ├── input/
│   └── output/
├── explicit_fem/          # Explicit time stepping solver
│   ├── explicit_CH.py     # Main explicit solver
│   ├── input/
│   └── output/
├── diff_fem/              # Differentiation-based (AD) solver
│   ├── diff_CH.py         # AD-based Cahn-Hilliard solver
│   ├── input/
│   └── output/
└── README.md
```

### Key Parameters in Configuration

All schemes share common parameters in `input/json/params.json`:

```json
{
  "dt": 5.0e-06,        // Time step size
  "t_OFF": 50.0e-06,    // Total simulation time
  "Lx": 1.0,            // Domain length in x
  "Ly": 1.0,            // Domain length in y
  "nx": 64,             // Number of elements in x
  "ny": 64,             // Number of elements in y
  "MnV": 1.0,           // M_c (mobility coefficient)
  "KnV": 1.0e-02,       // κ (gradient energy coefficient)
  "theta": 0.5          // θ parameter (0.5 = Crank-Nicolson)
}
```
---


## 6. References

1. Fenics document: https://olddocs.fenicsproject.org/dolfin/1.3.0/python/demo/documented/cahn-hilliard/python/documentation.html




