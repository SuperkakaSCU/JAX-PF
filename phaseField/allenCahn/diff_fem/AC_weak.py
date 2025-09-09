"""
AD treatment based on implicit finite element solver for Allen Cahn Equation
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob


from jax_fem.problem import Problem


from jax import config
config.update("jax_enable_x64", True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
 

class AllenCahn(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable - order parameter eta
        self.fe_p = self.fes[0]
        self.params = params

        MnV = self.params['MnV']
        KnV = self.params['KnV']

        MnV_gp = MnV*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))
        KnV_gp = KnV*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        self.internal_vars = [MnV_gp, KnV_gp]


    def get_universal_kernel(self):
        ### Hu: AD can be directly used here (jax.grad)
        def f_local_grad(p):
            return 4*p*(p-1.0)*(p-0.5)

        vmap_f_local_grad = jax.jit(jax.vmap(f_local_grad))


        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            """
            Handles the weak form with one cell.
            Assume trial function (p), test function (q)

            cell_sol_flat: (num_nodes*vec + ...,)
            cell_sol_list: [(num_nodes, vec), ...]
            x: (num_quads, dim)  ->  physical_quad_points
            cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_JxW: (num_vars, num_quads)
            cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            """

            #### Hu: Unassemble the values to different variables
            cell_sol_p_old, p_old, chi, MnV, KnV = cell_internal_vars

            print("cell_sol_p_old", cell_sol_p_old.shape)

            # print("MnV", MnV.shape)
            # jax.debug.print("MnV = {x}", x=MnV)

            # print("KnV", KnV.shape)
            # jax.debug.print("KnV = {x}", x=KnV)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_p = cell_sol_list[0]
            
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p = cell_shape_grads_list[0]

            
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p = cell_v_grads_JxW_list[0]

            cell_JxW_p = cell_JxW[0]
            

            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p_grads_old = np.sum(cell_sol_p_old[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            

            ######################
            ## Hu: Weak form
            ######################
            # Hu: Handle the term for gradient energy
            tmp1 = MnV[:, None, None] * KnV[:, None, None] * p_grads # (num_quads, vec_p, dim)
            # (num_quads, num_nodes_p, vec_p, dim) -> (num_nodes_p, vec_p)
            val1 = np.sum(tmp1[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))

            # Hu: Handle the term for free energy
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            tmp2 = MnV * vmap_f_local_grad(p) # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val2 = np.sum(tmp2[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)
            
            # Hu: Handle the term for time evolution
            dt = self.params['dt']
            tmp3 = (p - p_old)/dt # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val3 = np.sum(tmp3[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)


            weak_form = [val1 + val2 + val3] # [(num_nodes_p, vec_p)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel

    def set_initial_params(self, initial_params):
        # Override base class method.
        sol_p_old, noise = initial_params
        
        self.initial_internal_vars = [sol_p_old[self.fe_p.cells],
                              self.fe_p.convert_from_dof_to_quad(sol_p_old)[:, :, 0], 
                              np.repeat(noise[:, None], self.fe_p.num_quads, axis=1)]

        self.internal_vars = self.initial_internal_vars + self.internal_vars

    def set_params(self, params):
        self.internal_vars = params

