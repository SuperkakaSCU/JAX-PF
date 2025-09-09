import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse

from applications.phaseField.allenCahn.diff_fem.AC_weak import AllenCahn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from jax import config
config.update("jax_enable_x64", True)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')


def problem():
    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    # print("params", params)

    dt = params['dt']
    t_OFF = params['t_OFF']
    Lx = params['Lx']
    Ly = params['Ly']
    nx = params['nx']
    ny = params['ny']
    ele_type = 'QUAD4'  # polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    ## Hu: Sizes of domain
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])


    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def top(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], Ly, atol=1e-5)


    def neumann_val(point):
        return np.array([0.])



    ### Hu: Variable: [p]
    problem = AllenCahn(mesh, vec=1, dim=2, ele_type=ele_type, additional_info=[params])
    
    points = problem.fe_p.points

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
        [0.7, 0.95]])  

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

    
    ## Hu: Initial condition of eta
    sol_p = batched_scalar_IC(points).reshape(-1, 1) 


    sol_list = [sol_p]
    chi = jax.random.uniform(jax.random.PRNGKey(0), shape=(problem.fe_p.num_cells,)) - 0.5
    initial_params = [sol_list[0], chi]
    problem.set_initial_params([sol_list[0], chi])

    print("problem.internal_vars")
    print(type(problem.internal_vars))
    print(len(problem.internal_vars))

    # sol = np.zeros((problem.num_total_nodes, problem.vec))

    fwd_pred = ad_wrapper(problem)

    nIter = int(t_OFF/dt)

    def simulation(alpha):
        params = problem.internal_vars

        print("parmas", params[0].shape)
        print("parmas", params[1].shape)
        print("parmas", params[2].shape)
        print("parmas", params[3].shape)
        print("parmas", params[4].shape)

        coeff1, coeff2 = alpha

        # MnV
        params[3] = coeff1*params[3]
        # KnV
        params[4] = coeff2*params[4]

        
        for i in range(nIter):
            print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")

            sol_list = fwd_pred(params)

            sol = sol_list[0]
            params[0] = sol[problem.fe_p.cells]
            params[1] = problem.fe_p.convert_from_dof_to_quad(sol)[:, :, 0]

        area_fraction = np.sum(sol_list[0])

        return area_fraction


    # AD grads [102.502519  72.341267]
    # FDM: [102.51211722782045, 72.34007783949892]
        
    alpha = np.array([1.0, 2.0])
    grads = jax.grad(simulation)(alpha)
    

    
    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    obj1_upper = simulation(np.array([1.01, 2.0]))


    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    obj1_below = simulation(np.array([0.99, 2.0]))


    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    obj2_upper = simulation(np.array([1.00, 2.01]))


    problem.custom_init(params)
    problem.set_initial_params(initial_params)
    obj2_below = simulation(np.array([1.0, 1.99]))

    print("AD grads", grads)
    print("FDM: [{0}, {1}]".format((obj1_upper-obj1_below)/0.02, (obj2_upper-obj2_below)/0.02))


if __name__ == "__main__":
    problem()
    # plot_stress_strain()
    # plt.show()
