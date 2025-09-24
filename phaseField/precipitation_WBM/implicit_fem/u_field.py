"""
Implicit finite element solver for Momentum balance
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



class Inclusion(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_u = self.fes[0]

        ## Hu: elastic modulus
        self.C_Mg = onp.zeros((self.dim, self.dim, self.dim, self.dim))
        self.C_beta = onp.zeros((self.dim, self.dim, self.dim, self.dim))


        # isotropic elastic constants for Mg matrix
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


        self.C_Mg[0, 1, 0, 1] = C44
        self.C_Mg[0, 1, 1, 0] = C44
        self.C_Mg[1, 0, 0, 1] = C44
        self.C_Mg[1, 0, 1, 0] = C44

        
        # isotropic elastic constants for beta precipitate
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


        self.params = params



    ### Hu: nonExplicitEquationRHS
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    ## Hu: It has been distributed on each quad point
    def get_tensor_map(self):

        ### Hu: Interpolation function
        def h_local(n1, n2, n3):
            h1V = 10.0 * n1 * n1 * n1 - 15.0 * n1 * n1 * n1 * n1 + 6.0 * n1 * n1 * n1 * n1 * n1
            h2V = 10.0 * n2 * n2 * n2 - 15.0 * n2 * n2 * n2 * n2 + 6.0 * n2 * n2 * n2 * n2 * n2
            h3V = 10.0 * n3 * n3 * n3 - 15.0 * n3 * n3 * n3 * n3 + 6.0 * n3 * n3 * n3 * n3 * n3
            return h1V, h2V, h3V



        ## Hu: (u_grads, *internal_vars)
        def stress(u_grad, *internal_vars):
            # u_grad (3, 3)
            # sol_u_old (3,)
            # sol_c_old (1,)
            # sol_n1_old (1,)
            # c_old ()
            # n1_old ()

            sol_u_old, sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old, c_old, n1_old, n2_old, n3_old = internal_vars

            h1V, h2V, h3V = h_local(n1_old, n2_old, n3_old)

            CIJ_combined = self.C_Mg * (1.0 - h1V - h2V - h3V) + self.C_beta * (h1V + h2V + h3V)

            ## sfts = a_p * c_beta + b_p
            sfts1   = self.sfts_linear1 * c_old + self.sfts_const1
            sfts2   = self.sfts_linear2 * c_old + self.sfts_const2
            sfts3   = self.sfts_linear3 * c_old + self.sfts_const3


            epsilon = 0.5*(u_grad + u_grad.T)

            epsilon0 = sfts1*h1V + sfts2*h2V + sfts3*h3V

            E2 = epsilon - epsilon0

            sigma = np.sum(CIJ_combined[:, :, :, :] * E2[None, None, :, :], axis = (2, 3))
            

            return sigma
        return stress



    def set_params(self, params):
        # Override base class method.
        sol_u_old, sol_phaseField_list, quad_phaseField_old_list = params

        sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list

        quad_c_old, quad_n1_old, quad_n2_old, quad_n3_old = quad_phaseField_old_list

        self.internal_vars = [sol_u_old[self.fe_u.cells],
                              sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old,
                              quad_c_old, quad_n1_old, quad_n2_old, quad_n3_old]
