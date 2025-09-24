"""
Implicit finite element solver 2D for Cahn-Hilliard
Apply the theta-method to the mixed weak form of the equation
See https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_cahn-hilliard.html for details
"""
import jax
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob


from jax_fem.problem import Problem
from jax_fem import logger

from jax import config
config.update("jax_enable_x64", True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk/CH_theta')
os.makedirs(vtk_dir, exist_ok=True)

 

class CahnHilliard(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_p = self.fes[0]
        ## Hu: mu variable for forth order PDE
        self.fe_mu = self.fes[1]

        self.theta = 0.5

        self.params = params

        MnV = self.params['MnV']
        KnV = self.params['KnV']

        MnV_gp = MnV*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))
        KnV_gp = KnV*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        self.internal_vars = [MnV_gp, KnV_gp]


    def get_universal_kernel(self):
        ### TODO: AD-based f_local
        ## f,c
        def dfdc_func(c):
            df = 4.*(c-1.)*(c-0.5)*c
            return df

        vmap_dfdc_func = jax.jit(jax.vmap(dfdc_func))

        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            """
            Handles the weak form with one cell.
            Assume trial function (p, mu), test function (q, w)

            cell_sol_flat: (num_nodes*vec + ...,)
            cell_sol_list: [(num_nodes, vec), ...]
            x: (num_quads, dim)  ->  physical_quad_points
            cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_JxW: (num_vars, num_quads)
            cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            You may define fully implicit weak form, but that doesn't converge.
            Therefore, some of the terms are changed to explicit for good reason.
            In summary, only the "diffusion" like term is kept implict, and other terms are explicit.
            The entire weak form will be linear.
            """

            ##### p is phase field variable of 'concentration'

            #### Hu: Unassemble the values to different variables
            cell_sol_p_old, p_old, cell_sol_mu_old, mu_old, MnV, KnV = cell_internal_vars


            ## Hu: [p, mu] - [(num_nodes_p, vec_p), (num_nodes_mu, vec_mu)]
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_p, cell_sol_mu = cell_sol_list
            

            ## Hu: cell_shape_grads: (num_quads, num_nodes + ..., dim)
            ## Hu: cell_shape_grads_p: (num_quads, num_nodes, dim), cell_shape_grads_mu: (num_quads, num_nodes, dim)
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p, cell_shape_grads_mu = cell_shape_grads_list


            ## Hu: cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            ## Hu: cell_v_grads_JxW_p: (num_quads, num_nodes, 1, dim), cell_v_grads_JxW_mu: (num_quads, num_nodes, 1, dim)
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p, cell_v_grads_JxW_mu = cell_v_grads_JxW_list

            ## cell_JxW: (num_vars, num_quads)
            cell_JxW_p, cell_JxW_mu = cell_JxW[0], cell_JxW[1]
            

            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p_grads_old = np.sum(cell_sol_p_old[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)

            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            p_old = np.sum(cell_sol_p_old[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            mu_grads = np.sum(cell_sol_mu[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)
            mu_grads_old = np.sum(cell_sol_mu_old[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)

            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            mu_old = np.sum(cell_sol_mu_old[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
           


            ######################
            ## Hu: This is phase field variable weak form of p on each cell (Test function: q)
            ######################
            ## Hu: Handles the term `(p - p_old)*q*dx`
            dt = self.params['dt']
            tmp1 = (p - p_old)/dt # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val1 = np.sum(tmp1[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)
            
           
            ## Hu: Handles the term `inner(grad(mu), grad(q)*dx` [Mass Term]
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            # MnV = self.params['MnV']
            ## Hu: Theta-method with hybrid explici & implicit
            ## https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_cahn-hilliard.html
            cell_sol_mu_theta = (1.-self.theta)*cell_sol_mu_old + self.theta*cell_sol_mu  # (num_nodes_p, vec_mu) > (4, 1)
            mu_theta_grads = np.sum(cell_sol_mu_theta[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim) -> (4, 1, 2)
            tmp2 = MnV[:, None, None] * mu_theta_grads # (num_quads, vec_p, dim)

            ## Hu: (num_quads, 1, vec_mu, dim) * (num_quads, num_nodes_mu, 1, dim) -> 
            ## Hu: (num_quads, num_nodes_mu, vec_mu, dim) -> (num_nodes_mu, vec_mu)
            val2 = np.sum(tmp2[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))



            ######################
            ## Hu: This is phase field variable weak form of mu on each cell (Test function: w)
            ######################
            # Handles the term `mu*w*dx` [Left hand side]
            tmp3 = mu # (num_quads,)
            # (num_quads, 1, vec_mu) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_mu, vec_mu)
            val3 = np.sum(tmp3[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)


            ######################
            # Handles the term `dfdc*v*dx` [Left hand side] [Explicit]
            # tmp4 = vmap_dfdc_func(p) # (num_quads,)
            tmp4 = vmap_dfdc_func(p_old) # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_mu, vec_mu)
            val4 = -np.sum(tmp4[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)


            # Handles the term `inner(grad(p), grad(v)*dx` [Mass Term] [Implicit]
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            # KnV = self.params['KnV']
            tmp5 = KnV[:, None, None] * p_grads # (num_quads, vec_p, dim)
            

            # (num_quads, 1, vec_mu, dim) * (num_quads, num_nodes_mu, 1, dim)
            # (num_quads, num_nodes_mu, vec_mu, dim) -> (num_nodes_mu, vec_mu)
            val5 = -np.sum(tmp5[:, None, :, :] * cell_v_grads_JxW_mu, axis=(0, -1))


            # [sol_p, sol_mu]
            weak_form = [val1 + val2, val3 + val4 + val5] # [(num_nodes_p, vec_p), (num_nodes_mu, vec_mu)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel


    def set_params(self, params):
        self.internal_vars = params

    def set_initial_params(self, initial_params):
        # Override base class method.
        sol_p_old, sol_mu_old = initial_params

        ## [cell_sol_p_old, p_old, cell_sol_mu_old, mu_old]
        self.initial_internal_vars = [sol_p_old[self.fe_p.cells],
                              self.fe_p.convert_from_dof_to_quad(sol_p_old)[:, :, 0], 
                              sol_mu_old[self.fe_mu.cells],
                              self.fe_mu.convert_from_dof_to_quad(sol_mu_old)[:, :, 0]]

        self.internal_vars = self.initial_internal_vars + self.internal_vars

    def set_initial_guess(self, initial_sol):
        self.initial_guess = initial_sol
