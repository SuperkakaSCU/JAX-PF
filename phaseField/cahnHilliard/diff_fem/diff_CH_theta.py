"""
AD for difficientiable CH
Implicit finite element solver 2D for Cahn-Hilliard
Apply the theta-method to the mixed weak form of the equation
See https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_cahn-hilliard.html for details
"""
import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem import logger

from jax_fem.solver import implicit_vjp



from applications.phaseField.cahnHilliard.diff_fem.CH_weak import CahnHilliard


from jax import config
config.update("jax_enable_x64", True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        initial_guess = problem.initial_guess if hasattr(problem, 'initial_guess') else None
        sol_list = solver(problem, {'umfpack_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-9, 'line_search_flag': False})
        problem.set_initial_guess(sol_list)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options={'umfpack_solver':{}, 'initial_guess': sol_list, 'tol': 1e-9, 'line_search_flag': False})
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
# vtk_dir = os.path.join(output_dir, 'vtk/CH_theta')
# os.makedirs(vtk_dir, exist_ok=True)



def problem():
    json_file = os.path.join(input_dir, 'json/params_CH.json')
    params = json_parse(json_file)


    dt = params['dt']
    t_OFF = params['t_OFF']
    Lx = params['Lx']
    Ly = params['Ly']
    Lz = params['Lz']
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']

    # ele_type = 'TET10'  # polynomial of the element: 2
    ele_type = 'QUAD4'  # polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    print("Lx: {0}, Ly:{1}".format(Lx, Ly))

    
    def left(point):
        return np.isclose(point[0], 0., atol=1e-3)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-3)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-3)

    def back(point):
        return np.isclose(point[1], Ly, atol=1e-3)

    # def bottom(point):
    #     return np.isclose(point[2], 0., atol=1e-3)

    # def top(point):
    #     return np.isclose(point[2], Lz, atol=1e-3)

    

    ### Hu: [p, mu]
    problem = CahnHilliard(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type], \
                             additional_info=[params])
    


    ### Hu: Positions of fe_p
    points = problem.fe_p.points
    print("points", points.shape)
    
    
    
    ## Hu: Definition of initial condition
    ## Hu: 12 nucleation centers
    centers = np.array([
        [0.1, 0.3],
        [0.8, 0.7],
        [0.5, 0.2],
        [0.4, 0.4],
        [0.3, 0.9],
        [0.8, 0.1],
        [0.9, 0.5],
        [0.0, 0.1],
        [0.1, 0.6],
        [0.5, 0.6],
        [1.0, 1.0],
        [0.7, 0.95]
    ]) 


    ## Hu: Nucleation radius
    rads = np.array([12., 14., 19., 16., 11., 12., 17., 15., 20., 10., 11., 14.])

    domain_size = np.array([Lx, Ly]) 

    def scalar_IC_fn(p):
        scaled_centers = centers * domain_size 

        diff = p - scaled_centers             
        dists = np.linalg.norm(diff, axis=1)  
        
        contribs = 0.5 * (1.0 - np.tanh((dists - rads) / 1.5))  
        total = np.sum(contribs)
        return np.minimum(total, 1.0)  

    batched_scalar_IC = jax.vmap(scalar_IC_fn, in_axes=(0,))

    sol_p = batched_scalar_IC(points).reshape(-1, 1)  # shape: (N,)
    sol_mu = np.zeros((problem.fe_p.num_total_nodes, problem.fe_p.vec))
    sol_list = [sol_p, sol_mu]

    initial_params = [sol_list[0], sol_list[1]]
    problem.set_initial_params(initial_params)

    print("problem.internal_vars")
    print(type(problem.internal_vars))
    print(len(problem.internal_vars))

    problem.set_initial_guess(sol_list)
    fwd_pred = ad_wrapper(problem)


    nIter = int(t_OFF/dt)

    def simulation(alpha):
        params = problem.internal_vars

        # print("parmas", params[0].shape)
        # print("parmas", params[1].shape)
        # print("parmas", params[2].shape)
        # print("parmas", params[3].shape)
        # print("parmas", params[4].shape)

        coeff1, coeff2 = alpha

        # MnV
        params[4] = coeff1*params[4]
        # KnV
        params[5] = coeff2*params[5]

        for i in range(nIter + 1):
            print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
            
        
            sol_list = fwd_pred(params)
            problem.set_initial_guess(sol_list)

            # [sol_p, sol_mu]
            sol_p_old = sol_list[0]
            sol_mu_old = sol_list[1]

            params[0] = sol_p_old[problem.fe_p.cells]
            params[1] = problem.fe_p.convert_from_dof_to_quad(sol_p_old)[:, :, 0]
            params[2] = sol_mu_old[problem.fe_mu.cells]
            params[3] = problem.fe_mu.convert_from_dof_to_quad(sol_mu_old)[:, :, 0]

        

        area_fraction = np.sum(sol_list[0])
        return area_fraction
        

    # AD grads [2.735689 1.549975]
    # FDM: [2.735787809319845, 1.549971328108768]
    alpha = np.array([1.0, 2.0])
    grads = jax.grad(simulation)(alpha)
    
    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    problem.set_initial_guess(sol_list)
    obj1_upper = simulation(np.array([1.01, 2.0]))


    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    problem.set_initial_guess(sol_list)
    obj1_below = simulation(np.array([0.99, 2.0]))


    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    problem.set_initial_guess(sol_list)
    obj2_upper = simulation(np.array([1.00, 2.01]))


    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    problem.set_initial_guess(sol_list)
    obj2_below = simulation(np.array([1.0, 1.99]))

    print("AD grads", grads)
    print("FDM: [{0}, {1}]".format((obj1_upper-obj1_below)/0.02, (obj2_upper-obj2_below)/0.02))


if __name__ == '__main__':
    problem()
