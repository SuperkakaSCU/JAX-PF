"""
Implicit finite element solver for coupled Allen Cahn & Cahn Hilliard
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob
import scipy

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh, box_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem

from jax import config
config.update("jax_enable_x64", True)


onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)



# [Implicit]
class PhaseField(Problem):
    def custom_init(self, params):
        ### Hu: [c, mu, n1]
        ## Hu: phase field variable c
        self.fe_c = self.fes[0]
        # ## Hu: mu variable for forth order PDE
        # ## Hu: grad mu is expanded this time
        self.fe_mu = self.fes[1]
        ## Hu: phase field variable n1
        self.fe_n1 = self.fes[2]
        ## Hu: phase field variable n2
        self.fe_n2 = self.fes[3]
        ## Hu: phase field variable n1
        self.fe_n3 = self.fes[4]

        self.theta = 0.5

        
        ## Hu: elastic modulus
        self.C_Mg = onp.zeros((self.dim, self.dim, self.dim, self.dim))
        self.C_beta = onp.zeros((self.dim, self.dim, self.dim, self.dim))


        # isotropic elastic constants
        E = 40.0
        nu = 0.3
        C11 = E*(1-nu)/((1+nu)*(1-2*nu))
        C12 = E*nu/((1+nu)*(1-2*nu))
        C44 = E/(2.*(1. + nu))
        

        self.C_Mg[0, 0, 0, 0] = C11
        self.C_Mg[1, 1, 1, 1] = C11
        # self.C_Mg[2, 2, 2, 2] = C11

        self.C_Mg[0, 0, 1, 1] = C12
        self.C_Mg[1, 1, 0, 0] = C12

        # self.C_Mg[0, 0, 2, 2] = C12
        # self.C_Mg[2, 2, 0, 0] = C12

        # self.C_Mg[1, 1, 2, 2] = C12
        # self.C_Mg[2, 2, 1, 1] = C12

        # self.C_Mg[1, 2, 1, 2] = C44
        # self.C_Mg[1, 2, 2, 1] = C44
        # self.C_Mg[2, 1, 1, 2] = C44
        # self.C_Mg[2, 1, 2, 1] = C44

        # self.C_Mg[2, 0, 2, 0] = C44
        # self.C_Mg[2, 0, 0, 2] = C44
        # self.C_Mg[0, 2, 2, 0] = C44
        # self.C_Mg[0, 2, 0, 2] = C44

        self.C_Mg[0, 1, 0, 1] = C44
        self.C_Mg[0, 1, 1, 0] = C44
        self.C_Mg[1, 0, 0, 1] = C44
        self.C_Mg[1, 0, 1, 0] = C44

        

        E_beta = 50.0
        nu_beta = 0.3
        C11_beta = E_beta*(1-nu_beta)/((1+nu_beta)*(1-2*nu_beta))
        C12_beta = E_beta*nu_beta/((1+nu_beta)*(1-2*nu_beta))
        C44_beta = E_beta/(2.*(1. + nu_beta))
        

        self.C_beta[0, 0, 0, 0] = C11_beta
        self.C_beta[1, 1, 1, 1] = C11_beta
        # self.C_Mg[2, 2, 2, 2] = C11

        self.C_beta[0, 0, 1, 1] = C12_beta
        self.C_beta[1, 1, 0, 0] = C12_beta

        self.C_beta[0, 1, 0, 1] = C44_beta
        self.C_beta[0, 1, 1, 0] = C44_beta
        self.C_beta[1, 0, 0, 1] = C44_beta
        self.C_beta[1, 0, 1, 0] = C44_beta


        # A4, A3, A2, A1, and A0 Mg-Y matrix free energy parameters
        self.A4 = 1.3687
        self.A3 = -2.7375
        self.A2 = 5.1622
        self.A1 = -4.776
        self.A0 = -1.6704

        # B2, B1, and B0 Mg-Y matrix free energy parameters
        self.B2 = 5.0
        self.B1 = -5.9746
        self.B0 = -1.5924
                

        # The part of the stress free transformation strain proportional to the beta-phase composition
        # self.sfts_linear1 = np.array([[0., 0., 0.],
        #                               [0., 0., 0.],
        #                               [0., 0., 0.]])

        self.sfts_linear1 = np.array([[0., 0.],
                                      [0., 0.]])


        # The constant part of the stress free transformation strain
        # self.sfts_const1 = np.array([[0.0345, 0., 0.],
        #                              [0., 0.0185, 0.],
        #                              [0., 0., -0.00270]])

        self.sfts_const1 = np.array([[0.0345, 0.],
                                     [0., 0.0185]])

        # The part of the stress free transformation strain proportional to the beta-phase composition
        # self.sfts_linear2 = np.array([[0., 0., 0.],
        #                               [0., 0., 0.],
        #                               [0., 0., 0.]])
        self.sfts_linear2 = np.array([[0., 0.],
                                      [0., 0.]])


        # The constant part of the stress free transformation strain
        # self.sfts_const2 = np.array([[0.0225, -0.0069, 0.],
        #                              [-0.0069, 0.0305, 0.],
        #                              [0., 0., -0.00270]])
        self.sfts_const2 = np.array([[0.0225, -0.0069],
                                     [-0.0069, 0.0305]])


        # The part of the stress free transformation strain proportional to the beta-phase composition
        # self.sfts_linear3 = np.array([[0., 0., 0.],
        #                               [0., 0., 0.],
        #                               [0., 0., 0.]])
        self.sfts_linear3 = np.array([[0., 0.],
                                      [0., 0.]])


        # The constant part of the stress free transformation strain
        # self.sfts_const3 = np.array([[0.0225, 0.0069, 0.],
        #                              [0.0069, 0.0305, 0.],
        #                              [0., 0., -0.00270]])
        self.sfts_const3 = np.array([[0.0225, 0.0069],
                                     [0.0069, 0.0305]])




        # Kn1 the gradient energy coefficient
        # self.Kn1V = np.array([[0.03, 0., 0.],
        #                       [0., 0.007, 0.],
        #                       [0., 0., 1.0]])
        self.Kn1V = np.array([[0.03, 0.],
                              [0., 0.007]])

        # self.Kn2V = np.array([[0.01275, -0.009959, 0.],
        #                       [-0.009959, 0.02425, 0.],
        #                       [0., 0., 1.0]])
        self.Kn2V = np.array([[0.01275, -0.009959],
                              [-0.009959, 0.02425]])

        # self.Kn3V = np.array([[0.01275, 0.009959, 0.],
        #                       [0.009959, 0.02425, 0.],
        #                       [0., 0., 1.0]])
        self.Kn3V = np.array([[0.01275, 0.009959],
                              [0.009959, 0.02425]])

        self.params = params



    def get_universal_kernel(self):
        def f_local(c):
            faV = self.A0 + self.A1*c + self.A2*c*c + self.A3*c*c*c + self.A4*c*c*c*c
            fbV = self.B2*c*c + self.B1*c + self.B0
            return faV, fbV
        vmap_f_local = jax.vmap(f_local)

        def f_local_c(c):
            facV  = self.A1 + 2.0 * self.A2 * c + 3.0 * self.A3 * c * c + 4.0 * self.A4 * c * c * c
            fbcV  = 2.0 * self.B2 * c + self.B1

            faccV = 2.0 * self.A2 + 6.0 * self.A3 * c + 12.0 * self.A4 * c * c
            fbccV = 2.0 * self.B2
            return facV, fbcV, faccV, fbccV
        vmap_f_local_c = jax.vmap(f_local_c)


        ## Hu: Apply AD to get first derivative & second derivative
        ## Hu: AD could be appled to h_local, CIJ_combined_cal, sfts_cal using the following way
        # def vmap_f_local_c(c):
        #     grad_fa = jax.jit(jax.vmap(jax.grad(lambda c: f_local(c)[0])))
        #     grad_fb = jax.jit(jax.vmap(jax.grad(lambda c: f_local(c)[1])))
        #     grad_grad_fa = jax.jit(jax.vmap(jax.grad(jax.grad(lambda c: f_local(c)[0]))))
        #     grad_grad_fb = jax.jit(jax.vmap(jax.grad(jax.grad(lambda c: f_local(c)[1]))))
        #     facV = grad_fa(c)
        #     fbcV = grad_fb(c)
        #     faccV = grad_grad_fa(c)
        #     fbccV = grad_grad_fb(c)
        #     return facV, fbcV, faccV, fbccV
        
        
        ### Hu: Interpolation function
        def h_local(n1, n2, n3):
            h1V = 10.0 * n1 * n1 * n1 - 15.0 * n1 * n1 * n1 * n1 + 6.0 * n1 * n1 * n1 * n1 * n1
            h2V = 10.0 * n2 * n2 * n2 - 15.0 * n2 * n2 * n2 * n2 + 6.0 * n2 * n2 * n2 * n2 * n2
            h3V = 10.0 * n3 * n3 * n3 - 15.0 * n3 * n3 * n3 * n3 + 6.0 * n3 * n3 * n3 * n3 * n3
            return h1V, h2V, h3V
        vmap_h_local = jax.vmap(h_local, in_axes=(0, 0, 0))


        # first derivative
        def h_local_n1(n1, n2, n3):
            hn1V = 30.0 * n1 * n1 - 60.0 * n1 * n1 * n1 + 30.0 * n1 * n1 * n1 * n1
            hn2V = 30.0 * n2 * n2 - 60.0 * n2 * n2 * n2 + 30.0 * n2 * n2 * n2 * n2
            hn3V = 30.0 * n3 * n3 - 60.0 * n3 * n3 * n3 + 30.0 * n3 * n3 * n3 * n3
            return hn1V, hn2V, hn3V
        vmap_h_local_n1 = jax.vmap(h_local_n1, in_axes=(0, 0, 0))


        ### Hu: Calculate Elastic Modulus
        def CIJ_combined_cal(h1V, h2V, h3V):
            CIJ_combined = self.C_Mg * (1.0 - h1V - h2V - h3V) + self.C_beta * (h1V + h2V + h3V)
            return CIJ_combined
        vmap_CIJ_combined_cal = jax.vmap(CIJ_combined_cal, in_axes=(0, 0, 0))


        ### Hu: Calculate sfts1
        def sfts_cal(c, sfts_linear, sfts_const):
            sfts  = sfts_linear * c + sfts_const
            return sfts
        vmap_sfts_cal = jax.vmap(sfts_cal, in_axes=(0, None, None))


        ### Hu: Calculate sfts1c
        def sftsc_cal(c, sfts_linear, sfts_const):
            sftsc = sfts_linear
            return sftsc
        vmap_sftsc_cal = jax.vmap(sftsc_cal, in_axes=(0, None, None))


        def sftscc_cal(c, sfts_linear, sfts_const):
            sftscc = 0.0
            return sftscc
        vmap_sftscc_cal = jax.vmap(sftscc_cal, in_axes=(0, None, None))



        ### Hu: Calculate epsilon = 0.5 * (u_grad + u_grad.T)
        def epsilon_cal(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            return epsilon
        vmap_epsilon_cal = jax.vmap(epsilon_cal)


        def epsilon0_cal(sfts1, sfts2, sfts3, h1V, h2V, h3V):
            return sfts1*h1V + sfts2*h2V + sfts3*h3V
        vmap_epsilon0_cal = jax.vmap(epsilon0_cal, in_axes=(0, 0, 0, 0, 0, 0))



        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            ## TODO: some terms are not useful
            cell_sol_c_old, c_old, cell_sol_mu_old, mu_old, cell_sol_n1_old, n1_old, cell_sol_n2_old, n2_old, cell_sol_n3_old, n3_old, cell_sol_u_old, u_old = cell_internal_vars
            
            print("cell_sol_n1_old", cell_sol_n1_old.shape)

            #### Hu: Unassemble the values to different variables
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            ## (num_nodes_p, vec_p)
            cell_sol_c, cell_sol_mu, cell_sol_n1, cell_sol_n2, cell_sol_n3 = cell_sol_list
            

            ## Hu: cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            ## Hu: cell_shape_grads_p: (num_quads, num_nodes, dim)
            cell_shape_grads_c, cell_shape_grads_mu, cell_shape_grads_n1, cell_shape_grads_n2, cell_shape_grads_n3 = cell_shape_grads_list


            ## Hu: cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            ## Hu: cell_v_grads_JxW_p: (num_quads, num_nodes, 1, dim)
            cell_v_grads_JxW_c, cell_v_grads_JxW_mu, cell_v_grads_JxW_n1, cell_v_grads_JxW_n2, cell_v_grads_JxW_n3 = cell_v_grads_JxW_list


            ## Hu: cell_JxW: (num_vars, num_quads)
            cell_JxW_c, cell_JxW_mu, cell_JxW_n1, cell_JxW_n2, cell_JxW_n3 = cell_JxW[0], cell_JxW[1], cell_JxW[2], cell_JxW[3], cell_JxW[4]



            # Handles the term `inner(..., grad(p)*dx` [Hybrid implicit/explicit]
            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            c_grads = np.sum(cell_sol_c[None, :, :, None] * cell_shape_grads_c[:, :, None, :], axis=1) # (num_quads, vec_c, dim)
            c_grads_old = np.sum(cell_sol_c_old[None, :, :, None] * cell_shape_grads_c[:, :, None, :], axis=1) # (num_quads, vec_c, dim)
            c = np.sum(cell_sol_c[None, :, :] * self.fe_c.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            c_old = np.sum(cell_sol_c_old[None, :, :] * self.fe_c.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            mu_grads = np.sum(cell_sol_mu[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)
            mu_grads_old = np.sum(cell_sol_mu_old[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)
            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            mu_old = np.sum(cell_sol_mu_old[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            n1_grads = np.sum(cell_sol_n1[None, :, :, None] * cell_shape_grads_n1[:, :, None, :], axis=1) # (num_quads, vec_n1, dim)
            n1_grads_old = np.sum(cell_sol_n1_old[None, :, :, None] * cell_shape_grads_n1[:, :, None, :], axis=1) # (num_quads, vec_n1, dim)
            n1 = np.sum(cell_sol_n1[None, :, :] * self.fe_n1.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n1_old = np.sum(cell_sol_n1_old[None, :, :] * self.fe_n1.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            n2_grads = np.sum(cell_sol_n2[None, :, :, None] * cell_shape_grads_n2[:, :, None, :], axis=1)
            n2_grads_old = np.sum(cell_sol_n2_old[None, :, :, None] * cell_shape_grads_n2[:, :, None, :], axis=1)
            n2 = np.sum(cell_sol_n2[None, :, :] * self.fe_n2.shape_vals[:, :, None], axis=1)[:, 0]
            n2_old = np.sum(cell_sol_n2_old[None, :, :] * self.fe_n2.shape_vals[:, :, None], axis=1)[:, 0]

            n3_grads = np.sum(cell_sol_n3[None, :, :, None] * cell_shape_grads_n3[:, :, None, :], axis=1)
            n3_grads_old = np.sum(cell_sol_n3_old[None, :, :, None] * cell_shape_grads_n3[:, :, None, :], axis=1)
            n3 = np.sum(cell_sol_n3[None, :, :] * self.fe_n3.shape_vals[:, :, None], axis=1)[:, 0]
            n3_old = np.sum(cell_sol_n3_old[None, :, :] * self.fe_n3.shape_vals[:, :, None], axis=1)[:, 0]

            ### TODO: if they have different mesh?
            u_grads = np.sum(cell_sol_u_old[None, :, :, None] * cell_shape_grads_c[:, :, None, :], axis=1) # (num_quads, vec_u, dim)
            

            h1V, h2V, h3V = vmap_h_local(n1, n2, n3)
            hn1V, hn2V, hn3V = vmap_h_local_n1(n1, n2, n3)
            

            faV, fbV = vmap_f_local(c)
            facV, fbcV, faccV, fbccV = vmap_f_local_c(c)
            
            #### TODO: vvmap for further acceleration
            sfts1 = vmap_sfts_cal(c, self.sfts_linear1, self.sfts_const1)
            sfts2 = vmap_sfts_cal(c, self.sfts_linear2, self.sfts_const2)
            sfts3 = vmap_sfts_cal(c, self.sfts_linear3, self.sfts_const3)

            sfts1c = vmap_sftsc_cal(c, self.sfts_linear1, self.sfts_const1)
            sfts2c = vmap_sftsc_cal(c, self.sfts_linear2, self.sfts_const2)
            sfts3c = vmap_sftsc_cal(c, self.sfts_linear3, self.sfts_const3)

            sfts1cc = vmap_sftscc_cal(c, self.sfts_linear1, self.sfts_const1)
            sfts2cc = vmap_sftscc_cal(c, self.sfts_linear2, self.sfts_const2)
            sfts3cc = vmap_sftscc_cal(c, self.sfts_linear3, self.sfts_const3) # (num_quad,)

        
            dt = self.params['dt']


            ######################
            ######################
            ## This is phase field variable weak form of n1, n2, and n3
            ######################
            ######################
            # Handles the term `(n - n_old)*q*dx` [Left hand side]
            tmp_dn1 = (n1 - n1_old)/dt # (num_quads,)
            tmp_dn2 = (n2 - n2_old)/dt
            tmp_dn3 = (n3 - n3_old)/dt

            # (num_nodes, 1)
            val_dn1 = np.sum(tmp_dn1[:, None, None] * self.fe_n1.shape_vals[:, :, None] * cell_JxW_n1[:, None, None], axis=0)
            val_dn2 = np.sum(tmp_dn2[:, None, None] * self.fe_n2.shape_vals[:, :, None] * cell_JxW_n2[:, None, None], axis=0)
            val_dn3 = np.sum(tmp_dn3[:, None, None] * self.fe_n3.shape_vals[:, :, None] * cell_JxW_n3[:, None, None], axis=0)
            

            ### First term about (fbV - faV) * hn1V
            tmp_n1_1 = (fbV - faV) * hn1V # (num_quads,) 
            tmp_n2_1 = (fbV - faV) * hn2V # (num_quads,) 
            tmp_n3_1 = (fbV - faV) * hn3V # (num_quads,) 


            ### Second term about nDependentMisfitACp = -C*(E-E0)*(E0_p*Hn)
            CIJ_combined = vmap_CIJ_combined_cal(h1V, h2V, h3V)

            # (num_quads, vec_p, dim)
            epsilon = vmap_epsilon_cal(u_grads)
            epsilon0 = vmap_epsilon0_cal(sfts1, sfts2, sfts3, h1V, h2V, h3V)
            
            E2 = epsilon - epsilon0

            ### (num_quads, i, j, k, l) * (num_quads, None, None, k, l) -> (num_quads, i, j)
            sigma_ctr = np.sum(CIJ_combined[:, :, :, :, :] * E2[:, None, None, :, :], axis = (3, 4))

            # Compute one of the stress terms in the order parameter chemical potential,
            # nDependentMisfitACp = -C*(E-E0)*(E0_p*Hn)
            ## nDependentMisfitAC1 (num_quad,)
            nDependentMisfitAC1 = -np.sum(sigma_ctr[:, :, :]*sfts1[:, :, :], axis=(1, 2))
            nDependentMisfitAC2 = -np.sum(sigma_ctr[:, :, :]*sfts2[:, :, :], axis=(1, 2))
            nDependentMisfitAC3 = -np.sum(sigma_ctr[:, :, :]*sfts3[:, :, :], axis=(1, 2))
            
            nDependentMisfitAC1 = nDependentMisfitAC1*hn1V
            nDependentMisfitAC2 = nDependentMisfitAC2*hn2V
            nDependentMisfitAC3 = nDependentMisfitAC3*hn3V
            
            tmp_n1_2 = nDependentMisfitAC1
            tmp_n2_2 = nDependentMisfitAC2
            tmp_n3_2 = nDependentMisfitAC3
            

            ### Third term about heterMechACp = 0.5*Hn*(C_beta-C_alpha)*(E-E0)*(E-E0)
            # Compute the other stress term in the order parameter chemical potential,
            # heterMechACp = 0.5*Hn*(C_beta-C_alpha)*(E-E0)*(E-E0)
            ### (num_quads, i, j, k, l) * (num_quads, None, None, k, l) -> (num_quads, i, j)
            heterMechAC1 = np.sum((self.C_beta - self.C_Mg)[None, :, :, :, :] * E2[:, None, None, :, :], axis = (3, 4))
            tmp_n1_3 = np.sum(heterMechAC1*E2, axis = (1, 2)) # (num_quads,)

            tmp_n1_3 = +0.5*hn1V*tmp_n1_3
            tmp_n2_3 = +0.5*hn2V*tmp_n1_3
            tmp_n3_3 = +0.5*hn3V*tmp_n1_3

            Mn1V = self.params['Mn1V']
            Mn2V = self.params['Mn2V']
            Mn3V = self.params['Mn3V']

            tmp_n1 = Mn1V*(tmp_n1_1 + tmp_n1_2 + tmp_n1_3)
            tmp_n2 = Mn2V*(tmp_n2_1 + tmp_n2_2 + tmp_n2_3)
            tmp_n3 = Mn3V*(tmp_n3_1 + tmp_n3_2 + tmp_n3_3)

            # (num_nodes, 1)
            val_n1 = np.sum(tmp_n1[:, None, None] * self.fe_n1.shape_vals[:, :, None] * cell_JxW_n1[:, None, None], axis=0)
            val_n2 = np.sum(tmp_n2[:, None, None] * self.fe_n2.shape_vals[:, :, None] * cell_JxW_n2[:, None, None], axis=0)
            val_n3 = np.sum(tmp_n3[:, None, None] * self.fe_n3.shape_vals[:, :, None] * cell_JxW_n3[:, None, None], axis=0)


            # Handles the term `inner(grad(p), grad(v)*dx` [Mass Term]
            ## Hu: (None, None, i, j) * (num_quads, vec_p, None, dim) -> (num_quads, vec_p, dim)
            tmp_n1_4 = np.sum(self.Kn1V[None, None, :, :] * n1_grads[:, :, None, :], axis=-1) # (num_quads, vec_p, dim)
            tmp_n2_4 = np.sum(self.Kn2V[None, None, :, :] * n2_grads[:, :, None, :], axis=-1) # (num_quads, vec_p, dim)
            tmp_n3_4 = np.sum(self.Kn3V[None, None, :, :] * n3_grads[:, :, None, :], axis=-1) # (num_quads, vec_p, dim)
            
            
            ## Hu: val2 = ∑_q [ (∇c_j(q) ⋅ ∇v_j(q)) * JxW_v]
            # (num_quads, 1, vec_mu, dim) * (num_quads, num_nodes_mu, 1, dim)
            # (num_quads, num_nodes_mu, vec_mu, dim) -> (num_nodes_mu, vec_mu)
            val_n1_x = np.sum(tmp_n1_4[:, None, :, :] * cell_v_grads_JxW_n1, axis=(0, -1))
            val_n2_x = np.sum(tmp_n2_4[:, None, :, :] * cell_v_grads_JxW_n2, axis=(0, -1))
            val_n3_x = np.sum(tmp_n3_4[:, None, :, :] * cell_v_grads_JxW_n3, axis=(0, -1))

            val_n1_x = val_n1_x*Mn1V
            val_n2_x = val_n2_x*Mn2V
            val_n3_x = val_n3_x*Mn3V


            ######################
            ######################
            ## This is phase field variable weak form of c
            ######################
            ######################
            #### Hu: Definition of weak form on each cell
            # Handles the term `(c - c_old)*q*dx` [Left hand side]
            tmp_dc = (c - c_old)/dt # (num_quads,)
            val_dc = np.sum(tmp_dc[:, None, None] * self.fe_c.shape_vals[:, :, None] * cell_JxW_c[:, None, None], axis=0)

            # Handles the term `inner(grad(mu), grad(q)*dx` [Mass Term]
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            McV = self.params['McV']
            cell_sol_mu_theta = (1.-self.theta)*cell_sol_mu_old + self.theta*cell_sol_mu  # (num_nodes_p, vec_mu) 
            mu_theta_grads = np.sum(cell_sol_mu_theta[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim) 

            # tmp_c = McV * fcc_inv[:, None, None] * mu_grads[:, :, :]
            tmp_c = McV * mu_theta_grads[:, :, :]
            
            # (num_quads, 1, vec_mu, dim) * (num_quads, num_nodes_mu, 1, dim)
            # (num_quads, num_nodes_mu, vec_mu, dim) -> (num_nodes_mu, vec_mu)
            val_c = np.sum(tmp_c[:, None, :, :] * cell_v_grads_JxW_c, axis=(0, -1))
            

            ######################
            ######################
            ## This is phase field variable weak form of mu
            ######################
            ######################
            tmp_dmu = mu # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val_dmu = np.sum(tmp_dmu[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)

            # Handles the term `dfdc*v*dx` [Right hand side]
            tmp_mu_1 = facV * (1.0 - h1V - h2V - h3V) + fbcV * (h1V + h2V + h3V)
            tmp_mu_1 = -tmp_mu_1

            epsilon_c = sfts1c*h1V[:, None, None] + sfts2c*h2V[:, None, None] + sfts3c*h3V[:, None, None]
            tmp_mu_2 = np.sum(sigma_ctr*epsilon_c, axis = (1, 2)) # (num_quads,)
            
            tmp_mu = tmp_mu_1 + tmp_mu_2
            

            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val_mu = np.sum(tmp_mu[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)


            # [sol_c, sol_mu, sol_n1, sol_n2, sol_n3]
            weak_form = [val_dc + val_c, val_dmu + val_mu, val_dn1 + val_n1 + val_n1_x, val_dn2 + val_n2 + val_n2_x, val_dn3 + val_n3 + val_n3_x] 

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel


    def set_params(self, params):
        # Override base class method.
        sol_phaseField_list, sol_u_for_PF_list = params

        sol_c_old, sol_mu_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list
        cell_sol_u_old, u_old = sol_u_for_PF_list

        
        ## [cell_sol_c_old, c_old, cell_sol_mu_old, mu_old, cell_sol_n1_old, n1_old, cell_sol_u_old, u_old]
        self.internal_vars = [sol_c_old[self.fe_c.cells],
                              self.fe_c.convert_from_dof_to_quad(sol_c_old)[:, :, 0], 
                              sol_mu_old[self.fe_mu.cells],
                              self.fe_mu.convert_from_dof_to_quad(sol_mu_old)[:, :, 0], 
                              sol_n1_old[self.fe_n1.cells],
                              self.fe_n1.convert_from_dof_to_quad(sol_n1_old)[:, :, 0],
                              sol_n2_old[self.fe_n2.cells],
                              self.fe_n2.convert_from_dof_to_quad(sol_n2_old)[:, :, 0],
                              sol_n3_old[self.fe_n3.cells],
                              self.fe_n3.convert_from_dof_to_quad(sol_n3_old)[:, :, 0],
                              cell_sol_u_old, 
                              u_old]






