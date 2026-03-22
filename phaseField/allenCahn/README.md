# Allen-Cahn Equation

## 1. Introduction

The Allen-Cahn equation is a fundamental partial differential equation in phase-field modeling, used to describe the evolution of non-conserved parameters.

### General Variational Form

The general Allen-Cahn equation is expressed in variational form as:

$$\frac{\partial \eta}{\partial t} = -M_{\eta} \frac{\delta F_{tot}}{\delta \eta}$$

where:
- $\eta$ is the order parameter
- $M_{\eta}$ is the mobility coefficient controlling the kinetics of evolution
- $F_{tot}$ is the total free energy functional, $\frac{\delta F_{tot}}{\delta \eta}$ is the functional derivative


### Total Free Energy

The total free energy functional consists of two main contributions:

$$F_{tot}(\eta) = \int_\Omega \left[f(\eta) + \frac{\kappa}{2}|\nabla\eta|^2\right] d\Omega$$

where:

1. **Chemical Free Energy**: $\int_\Omega f(\eta) d\Omega$
   - Double-well (or multi-well) form favors phase separation
   - The derivate of chemical potential (driving force) is computed as: $$f'(\eta) = \frac{\partial f}{\partial \eta} = 4\eta(\eta - 1)(\eta - 0.5)$$

2. **Interfacial Energy Density**: $\int_\Omega \frac{\kappa}{2}|\nabla\eta|^2 d\Omega$
   - Arising from the gradient energy terms
  
---

## 2. Time Discretization

### 2.1 Implicit Time Integration (Backward Euler)

**Discretization Scheme:**

$$\eta^{n+1} = \eta^n - \Delta t \cdot M_{\eta}\left( f'(\eta^{n+1}) - \kappa \Delta \eta^{n+1} \right)$$

Rearranging:

$$\frac{\eta^{n+1} - \eta^n}{\Delta t} - M\left(\kappa \Delta \eta^{n+1} - f'(\eta^{n+1})\right) = 0$$

**Characteristics:**
- Residual evaluated at the new time step $t^{n+1}$
- Allows larger time steps
- Requires solving a nonlinear system at each step
- Solved using Newton-Raphson method with automatic differentiation
- Implementation:  [implicit_fem/implicit_AC.py](implicit_fem/implicit_AC.py)
  

### 2.2 Explicit Time Integration

**Discretization Scheme:**

$$\eta^{n+1} = \eta^n - \Delta t \cdot M_{\eta}\left( f'(\eta^n) - \kappa \Delta \eta^n \right)$$

Rearranging:

$$\frac{\eta^{n+1}-\eta^n}{\Delta t}  = - M_{\eta}\left( f'(\eta^n) - \kappa \Delta \eta^n \right)$$

**Characteristics:**
- Evaluated at the current time step $t^n$
- Smaller time steps required
- Simpler per-step computation
- Implementation: [explicit_fem/explicit_AC.py](explicit_fem/explicit_AC.py)

---

## 3. Weak Formulation

### 3.1 Implicit Weak Form

**Variational Formulation:**

Find $\eta^{n+1} \in V$ such that for all test functions $q \in V$:

$$\int_\Omega \left[\frac{\eta^{n+1} - \eta^n}{\Delta t} q +  M_{\eta}\kappa \nabla\eta^{n+1} \cdot \nabla q +  M_{\eta}f'(\eta^{n+1}) q\right] d\Omega = 0$$

**Implementation Details:**

The weak form is assembled from three main contributions (from quadrature points defined by **tmp** to nodes defined by **val**):

1. **Gradient Energy Term:**
   ```
   val1 = ∫ M·κ ∇η ·∇q dΩ
   tmp1 = M * κ * η_grads
   val1 = ∑(tmp1 · cell_v_grads_JxW)
   ```

2. **Chemical Potential Term:**
   ```
   val2 = ∫ M·f'(η)·q dΩ
   tmp2 = M * vmap_f_local_grad(η)
   val2 = ∑(tmp2 · shape_vals · JxW)
   ```

3. **Time Evolution Term:**
   ```
   val3 = ∫ (ηⁿ⁺¹ - ηⁿ)/Δt · q dΩ
   tmp3 = (η - η_old) / dt
   val3 = ∑(tmp3 · shape_vals · JxW)
   ```

**Total Residual:**
$$R(\eta^{n+1}) = \text{val1} + \text{val2} + \text{val3} = 0$$

Solved using Newton's method with Jacobian computed via automatic differentiation.

### 3.2 Explicit Weak Form

In explicit schemes, the solution at the next time step is computed directly from the current state:

$$\eta^{n+1} = \eta^n - \Delta t  \mathbf{M}^{-1}\mathbf{r}$$

This method is straightforward to implement and computationally inexpensive per step. However, the stability condition requires the time step $\Delta t$ to be very small, especially for stiff systems such as PF models with fine spatial resolution or strong coupling between variables. Consequently, explicit schemes often demand a large number of time steps to reach physically meaningful time scales.

**Implementation Details:**

The explicit solver separates the computation into two independent problems:

#### Problem 1: **Mass Matrix** (LHS)

The **mass matrix** **$\mathbf{M}$** is assembled from the weak form and then inverted (as diagonal inverse for efficiency) to compute **$\mathbf{M}^{-1}$**.

#### Problem 2: **Force Vector** (RHS)

The **force vector** consists of two contributions:

1. **Gradient Energy Term:** `tmp1 = M_η * κ * η_grads`
2. **Chemical Potential Term:** `tmp2 = M_η * f'(η)`



## 4. Implementation Schemes

### Directory Structure

```
allenCahn/
├── implicit_fem/          # Implicit backward Euler solver
│   ├── implicit_AC.py     # Main implicit solver
│   ├── input/
│   └── output/
├── explicit_fem/          # Explicit forward Euler solver
│   ├── explicit_AC.py     # Main explicit solver
│   ├── input/
│   └── output/
├── diff_fem/              # Differentiation-based (AD) solver
│   ├── AC_weak.py         # Weak form definition
│   ├── diff_AC.py         # Main AD solver
│   ├── input/
│   └── output/
└── README.md
```

### Key Parameters in Configuration

All schemes share common parameters in `input/json/params.json`:

```json
{
  "dt": 0.01,           // Time step size
  "t_OFF": 10.0,        // Total simulation time
  "Lx": 1.0,            // Domain length in x
  "Ly": 1.0,            // Domain length in y
  "nx": 64,             // Number of elements in x
  "ny": 64,             // Number of elements in y
  "MnV": 0.1,           // M_eta (mobility coefficient)
  "KnV": 0.01           // kappa (interface parameter)
}
```

## 4. Typical Applications
- Crystal growth and solidification
- Grain coarsening
- Crack propagation in brittle materials
- Recrystallization dynamics



