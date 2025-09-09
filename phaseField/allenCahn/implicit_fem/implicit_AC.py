"""
Implicit finite element solver for Allen Cahn Equation
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
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

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         return np.array([0.])
    #     return [surface_map, surface_map, surface_map, surface_map]


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
            cell_sol_p_old, p_old, chi = cell_internal_vars

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
            MnV = self.params['MnV']
            KnV = self.params['KnV']

            # Hu: Handle the term for gradient energy
            tmp1 = MnV * KnV * p_grads # (num_quads, vec_p, dim)
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

    def set_params(self, params):
        # Override base class method.
        sol_p_old, noise = params
        print(type(sol_p_old))
        print("sol_p_old", sol_p_old.shape)
        self.internal_vars = [sol_p_old[self.fe_p.cells],
                              self.fe_p.convert_from_dof_to_quad(sol_p_old)[:, :, 0], 
                              np.repeat(noise[:, None], self.fe_p.num_quads, axis=1)]


def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:06d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

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
    # print("points", points.shape)

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
    save_sols(problem, sol_list, 0)
    
    chi = jax.random.uniform(jax.random.PRNGKey(0), shape=(problem.fe_p.num_cells,)) - 0.5


    nIter = int(t_OFF/dt)

    for i in range(nIter + 1):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
        
        problem.set_params([sol_list[0], chi])

        sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list})   

        # if (i + 1) % 5 == 0:
        #     save_sols(problem, sol_list, i + 1)
        save_sols(problem, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is implicit solver for Allen-Cahn benchmark:", end_time - start_time)
