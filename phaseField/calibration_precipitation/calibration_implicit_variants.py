"""
Calibration
Implicit finite element solver for coupled Allen Cahn & Cahn Hilliard & Momentum balance
Mg precipitate of single varient considering WBM model
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as np
import jax.flatten_util
from jax import config

import numpy as onp

import meshio
import sys
import glob
import scipy

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh, box_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem
from jax_fem import logger
from jax_fem.solver import implicit_vjp

## Hu: For gradient-based optimization
## See https://scipy.org/ for more information
import scipy
from scipy.optimize import minimize

## Hu: Weak form for [c, mu, n1]
from applications.phaseField.calibration_precipitation.phase_field import PhaseField
## Hu: Weak form solver for [u]
from applications.phaseField.calibration_precipitation.u_field import Inclusion



config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
calibration_dir = os.path.join(output_dir, 'calibration')
os.makedirs(calibration_dir, exist_ok=True)


alpha_dir = os.path.join(calibration_dir, f'alpha.txt')
obj_dir = os.path.join(calibration_dir, f'obj_val.txt')
iteration_dir = os.path.join(calibration_dir, f'iteration.txt')
vtk_ref_dir = os.path.join(input_dir, f'vtk_ref_1000steps/p_001000.vtu')

npy_c_dir = os.path.join(input_dir, f'vtk_ref_1000steps/sol_c_final.npy')
npy_n1_dir = os.path.join(input_dir, f'vtk_ref_1000steps/sol_n1_final.npy')
npy_n2_dir = os.path.join(input_dir, f'vtk_ref_1000steps/sol_n2_final.npy')
npy_n3_dir = os.path.join(input_dir, f'vtk_ref_1000steps/sol_n3_final.npy')


## Hu: Automatic differentiation wrapper for the forward problem.
## See https://doi.org/10.1016/j.cpc.2023.108802 for details
def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    """
    Parameters
    ----------
    problem : Problem
    solver_options : dictionary
    adjoint_solver_options : dictionary

    Returns
    -------
    fwd_pred : callable
    """
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        initial_guess = problem.initial_guess if hasattr(problem, 'initial_guess') else None
        sol_list = solver(problem, {'spsolve_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-7, 'line_search_flag': False})
        problem.set_initial_guess(sol_list)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options={'spsolve_solver':{}, 'initial_guess': sol_list, 'tol': 1e-7, 'line_search_flag': False})
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred


def compute_max_op_from_sol_list_jax(sol_list, threshold=0.01):
    sol_list = [np.squeeze(s) for s in sol_list]
    op_array = np.stack(sol_list, axis=1)

    max_op_idx = np.argmax(op_array, axis=1)
    op_sum = np.sum(op_array, axis=1)
    max_op_idx = np.where(op_sum < threshold, -1, max_op_idx)

    return max_op_idx



def problem():
    json_file = os.path.join(input_dir, 'json/params_KKS.json')
    params = json_parse(json_file)


    dt = params['dt']
    t_OFF = params['t_OFF']
    Lx = params['Lx']
    Ly = params['Ly']
    Lz = params['Lz']
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']


    ele_type = 'QUAD4'  # 2D polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    ## Hu: Sizes of domain 
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    print("Lx: {0}, Ly:{1}".format(Lx, Ly)) 

    
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)


    ## Hu: Define Dirichlet B.C. for [u]
    def zero_dirichlet_val(point):
        return 0.


    dirichlet_bc_info_u = [[left, right, bottom, top, left, right, bottom, top],
                         [0, 0, 0, 0, 1, 1, 1, 1],
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, \
                          zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]

    num_order = 3

    ### Hu: Define PF problem on top of JAX-FEM
    ### Hu: [c, mu, n1, n2, n3]
    phaseField = PhaseField(mesh=[mesh, mesh, mesh, mesh, mesh], vec=[1, 1, 1, 1, 1], dim=2, ele_type=[ele_type, ele_type, ele_type, ele_type, ele_type],  additional_info=[params])

    ### Hu: Define Eshelby inclusion problem on top of JAX-FEM
    ### Hu: [u] - [ux, uy, uz]
    problem_u = Inclusion(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info_u, additional_info=[params])
    
    ### Hu: Positions of [c, mu, n1, n2, n3]
    points = phaseField.fe_c.points
    

    ### Hu: set initial conditions for phaseField [c, n1, n2, n3]
    scalar_c_IC = np.ones((phaseField.fes[0].num_total_nodes, phaseField.fes[0].vec))*0.04

    ### Hu: IC for mu
    scalar_mu_IC = np.zeros((phaseField.fes[1].num_total_nodes, phaseField.fes[1].vec))
    
    ### Hu: IC for n1, n2, n3
    def set_ICs(sol_IC, p, domain_size, index, center, rad):        
        p = p.reshape(nx+1, ny+1, phaseField.fes[1].dim)

        scaled_centers = center * domain_size

        dist = np.linalg.norm(p - scaled_centers, axis=2)  
        
        sol_IC = sol_IC.at[index].set(0.5 * (1.0 - np.tanh((dist - rad) / (Lx/nx))))
       
        return sol_IC


    ## Work on each nucleation
    vmap_set_ICs = jax.jit(jax.vmap(set_ICs, in_axes=(None, None, None, 0, 0, 0)))

    ## Initialization of 3 order params (0~2)
    # (num_order, nx, ny)
    sol_IC = np.array([np.zeros((nx+1, ny+1))] * 3)
    index_list  = np.array([0, 0, 1, 2])

    ## Initialization of nucleations for 4 order params 
    # 15 centers, normalized coordinates (0~1)
    # (num_nucli, 2)
    center_list = np.array([[1.0 / 3.0, 1.0 / 3.0], 
                            [2.0 / 3.0, 2.0 / 3.0], 
                            [3.0 / 4.0, 1.0 / 4.0], 
                            [1.0 / 4.0, 3.0 / 4.0]])

    # (num_nucli, )
    rad_list = np.array([Lx/16.0]*4)

    domain_size = np.array([Lx, Ly]) 

    sol_IC_nucl = vmap_set_ICs(sol_IC, points, domain_size, index_list, center_list, rad_list)
    sol_IC_nucl = np.sum(sol_IC_nucl, axis=0)
    sol_IC_nucl = np.minimum(sol_IC_nucl, 0.999)
    sol_IC_nucl = sol_IC_nucl.reshape(num_order, -1)
    

    ## phaseField [c, n1, n2, n3]
    sol_phaseField_list = [scalar_c_IC,
                           scalar_mu_IC,
                           sol_IC_nucl[0].reshape(phaseField.fe_n1.num_total_nodes, phaseField.fe_n1.vec), 
                           sol_IC_nucl[1].reshape(phaseField.fe_n2.num_total_nodes, phaseField.fe_n2.vec),
                           sol_IC_nucl[2].reshape(phaseField.fe_n3.num_total_nodes, phaseField.fe_n3.vec)]

    problem_u.IC_phaseField_list = sol_phaseField_list
    
    ### Hu: pseudo IC for displacement u
    sol_u_list = [np.zeros((problem_u.fes[0].num_total_nodes, problem_u.fes[0].vec))]


    ############
    ## Hu: Pass pseudo ICs of u to phase field solver
    ############
    sol_u_old = sol_u_list[0]
    sol_u_for_PF_list = [sol_u_old[problem_u.fe_u.cells], problem_u.fe_u.convert_from_dof_to_quad(sol_u_old)]
        
    ## Store initial conditions
    sol_IC_pf_u_list = [sol_phaseField_list, sol_u_list]
    sol_IC_list = [sol_phaseField_list, sol_u_for_PF_list]



    ############
    ## Hu: Set initial params for both problem
    ############
    sol_c_old, sol_mu_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list
        
    sol_phaseField_old_list = [sol_c_old[phaseField.fe_c.cells],
                               sol_n1_old[phaseField.fe_n1.cells], sol_n2_old[phaseField.fe_n2.cells], sol_n3_old[phaseField.fe_n3.cells]]                      
    
    quad_phaseField_old_list = [phaseField.fe_c.convert_from_dof_to_quad(sol_c_old)[:, :, 0],
                                phaseField.fe_n1.convert_from_dof_to_quad(sol_n1_old)[:, :, 0],
                                phaseField.fe_n2.convert_from_dof_to_quad(sol_n2_old)[:, :, 0],
                                phaseField.fe_n3.convert_from_dof_to_quad(sol_n3_old)[:, :, 0],]

    initial_phaseField_params = [sol_u_list[0], sol_phaseField_old_list, quad_phaseField_old_list]
    problem_u.set_initial_params(initial_phaseField_params)
    problem_u.set_initial_guess(sol_u_list)
    fwd_pred_u = ad_wrapper(problem_u)


    ## Set for phaseField
    sol_u_old = sol_u_list[0]
    sol_u_for_PF_list = [sol_u_old[problem_u.fe_u.cells], problem_u.fe_u.convert_from_dof_to_quad(sol_u_old)]

    initial_u_params = [sol_phaseField_list, sol_u_for_PF_list]
    phaseField.set_initial_params(initial_u_params)
    phaseField.set_initial_guess(sol_phaseField_list)

    fwd_pred_pf = ad_wrapper(phaseField)


    ############
    ## Hu: Read the npy file for reference results
    ############
    c_ref = onp.load(npy_c_dir)       # scalar field at points
    n1_ref = onp.load(npy_n1_dir) 
    n2_ref = onp.load(npy_n2_dir) 
    n3_ref = onp.load(npy_n3_dir) 

    # (num_nodes, 1)
    c_ref = np.array(c_ref)
    n1_ref = np.array(n1_ref)
    n2_ref = np.array(n2_ref)
    n3_ref = np.array(n3_ref)



    nIter = int(t_OFF/dt)

    def simulation(alpha):
        ## Hu: Calibration for [E, nu, E_beta, nu_beta, Mn1V, Mn2V, Mn3V, McV]
        # This is the key method for solving differentiable inverse problems.
        # We MUST define (override) 'set_params' method so that ``params`` become differentiable.
        # No need to define this method if only forward problem is solved.
        # See https://github.com/deepmodeling/jax-fem/blob/c3fbcb3ef9e44643a6afd59a20235a9def368c64/jax_fem/problem.py#L472

        coeff1, coeff2, coeff3, coeff4 , coeff5, coeff6, coeff7, coeff8= alpha

        params_u = problem_u.internal_vars
        params_pf = phaseField.internal_vars


        #### Set parameters for pf
        ## E
        params_pf[-8] = coeff1*params_pf[-8]
        ## nu
        params_pf[-7] = coeff2*params_pf[-7]
        ## E_beta
        params_pf[-6] = coeff3*params_pf[-6]
        ## nu_beata
        params_pf[-5] = coeff4*params_pf[-5]
        ## Mn1V
        params_pf[-4] = coeff5*params_pf[-4]
        ## Mn2V
        params_pf[-3] = coeff6*params_pf[-3]
        ## Mn3V
        params_pf[-2] = coeff7*params_pf[-2]
        ## MncV
        params_pf[-1] = coeff8*params_pf[-1]


        #### Set parameters for u
        ## E
        params_u[-4] = coeff1*params_u[-4]
        ## nu
        params_u[-3] = coeff2*params_u[-3]
        ## E_beta
        params_u[-2] = coeff3*params_u[-2]
        ## nu_beata
        params_u[-1] = coeff4*params_u[-1]


        sol_phaseField_list = problem_u.IC_phaseField_list
        
        for i in range(nIter + 1):
            print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")

            print(f"Start solving u at Step {i + 1} using implicit solver...")
            sol_u_list = fwd_pred_u(params_u)
            problem_u.internal_vars = params_u
            
            print(f"**Update PF result: [c, mu, n1, n2, n3] at Step {i + 1} based on Step {i + 1}'s u field**")
            print(f"Pass u at Step {i + 1} to [c, mu, n1, n2, n3]")
            sol_u_old = sol_u_list[0]
            sol_u_for_PF_list = [sol_u_old[problem_u.fe_u.cells], problem_u.fe_u.convert_from_dof_to_quad(sol_u_old)]

            ## Hu: Update self.internal variable
            phaseField.update_internal_params([sol_phaseField_list, sol_u_for_PF_list])
            params_pf = phaseField.internal_vars

            print(f"Solve [c, mu, n1, n2, n3] at Step {i + 1} using implicit solver based on [c, mu, n1, n2, n3] at Step {i} and [u] at Step {i + 1}...")
            sol_phaseField_list = fwd_pred_pf(params_pf)
            phaseField.internal_vars = params_pf


            print(f"**Update u at Step {i + 1} based on Step {i}'s PF result: [c, mu, n1, n2, n3]**")
            sol_c_old, sol_mu_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list
            
            sol_phaseField_old_list = [sol_c_old[phaseField.fe_c.cells],
                                       sol_n1_old[phaseField.fe_n1.cells], sol_n2_old[phaseField.fe_n2.cells], sol_n3_old[phaseField.fe_n3.cells]]                      
            
            quad_phaseField_old_list = [phaseField.fe_c.convert_from_dof_to_quad(sol_c_old)[:, :, 0],
                                        phaseField.fe_n1.convert_from_dof_to_quad(sol_n1_old)[:, :, 0],
                                        phaseField.fe_n2.convert_from_dof_to_quad(sol_n2_old)[:, :, 0],
                                        phaseField.fe_n3.convert_from_dof_to_quad(sol_n3_old)[:, :, 0],]

            #### TODO: sol_u_list[0] is not useful
            print("Pass [c, mu, n1, n2, n3] to u")
            problem_u.update_internal_params([sol_u_list[0], sol_phaseField_old_list, quad_phaseField_old_list])
            params_u = problem_u.internal_vars


        w = 1000.0

        obj1 = np.sum((sol_phaseField_list[2]-n1_ref)**2.0)
        obj2 = np.sum((sol_phaseField_list[3]-n2_ref)**2.0)
        obj3 = np.sum((sol_phaseField_list[4]-n3_ref)**2.0)

        obj = obj1+obj2+obj3
        obj = obj*w

        jax.debug.print("obj: {x}", x=obj)

        return obj
        # exit()


    alpha_write = open(alpha_dir, "w")
    alpha_write.close()

    obj_write = open(obj_dir, "w")
    obj_write.close()

    iteration_write = open(iteration_dir, "w")
    iteration_write.close()


    ### Hu: This new wrapper uses value_and_grad to increase efficiency
    ### Hu: Transfer jax.numpy to numpy -- objective functio
    def objective_wrapper(x):
        print("***Calling objective_wrapper***")
        print(f"x = {x}")
        x = np.array(x)

        sol_phaseField_list, sol_u_list = sol_IC_pf_u_list

        problem_u.set_initial_guess(sol_u_list)
        phaseField.set_initial_guess(sol_phaseField_list)

        problem_u.custom_init(params)
        phaseField.custom_init(params)

        problem_u.set_initial_params(initial_phaseField_params)
        phaseField.set_initial_params(initial_u_params)

        # ## Hu: initialize the global variables stored by JAX between calling forward CPFEM        
        obj_val, dJ = jax.value_and_grad(simulation)(x)
        objective_wrapper.dJ = dJ
        print(f"Finishes objective, obj_val = {obj_val}")

        obj_val = onp.array(obj_val)

        ## Hu: writing obj_val, time and alpha into file
        print("**Writing alpha into files**")
        alpha_write = open(alpha_dir, "a+")
        alpha_write.write(str(x))
        alpha_write.write('\n')
        alpha_write.close()

        print("**Writing obj value into files**")
        obj_write = open(obj_dir, "a+")
        obj_write.write(str(obj_val))
        obj_write.write('\n')
        obj_write.close()

        print("**Writing current time into files**")
        end_time_BFGS = time.time()
        run_time_BFGS = end_time_BFGS - start_time_BFGS
        iteration_write = open(iteration_dir, "a+")
        iteration_write.write(str(run_time_BFGS))
        iteration_write.write('\n')
        iteration_write.close()
        

        if obj_val < 1.0:
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            print("***Finishing Calibration***")
            #warnings.warn("Terminating optimization: iteration limit reached",TookTooManyIters)
            raise SystemExit(0)

        print("***Finishing objective***")
        return onp.array(obj_val, order='F', dtype=onp.float64)



    ## Hu: Define the derivative function
    def derivative_wrapper(x):
        print("***Calling derivative_wrapper***")
        x = np.array(x)
        grads = objective_wrapper.dJ

        print("***Finishing derivative***")

        # 'L-BFGS-B' & 'BFGS' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(grads, order='F', dtype=onp.float64)



    ## Hu: define callback for minimize()
    def callback(x):
        callback.nit += 1
        desired_iteration = 100 # for example you want it to stop after 10 iterations
    
        if callback.nit == desired_iteration:
            print("Final iterations: ", callback.nit)
            print("Final solution: ", x)
            end_time_BFGS = time.time()
            run_time_BFGS = end_time_BFGS - start_time_BFGS
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            raise StopIteration

        else:
            # you could print elapsed iterations, current solution
            # and current function value
            print("Elapsed iterations: ", callback.nit)
            print("Current solution: ", x)
            # print("Current function value: ", callback.fun(x))
    
    callback.nit = 0

    start_time_BFGS = time.time()

    ## Initial Guess
    pt = onp.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
    from scipy.optimize import Bounds
    print("***Perform the 'L-BFGS-B' algorithm search***")
    bounds = Bounds((0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7), (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5))
    alpha_result = minimize(objective_wrapper, pt, method='L-BFGS-B', bounds=bounds, jac=derivative_wrapper, callback = callback)


if __name__ == '__main__':
    import time
    start_time = time.time()
    problem()
    end_time = time.time()

    print("This is calibration based on implicit solver:", end_time - start_time)
