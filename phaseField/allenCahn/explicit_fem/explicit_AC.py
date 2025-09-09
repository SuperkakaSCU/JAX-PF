"""
Explicit finite element solver for Allen-Cahn Equation
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
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem

from jax import config
config.update("jax_enable_x64", True)

## Hu: Depend on GPU device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
 

# [LHS]
class AllenCahnMass(Problem):
    """
    If the problem structure is MU = F(U), this class handles the mass matrix M.
    """
    def custom_init(self, params):
        ## Hu: phase field variable - order parameter eta
        self.fe_p = self.fes[0]
        self.params = params


    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         return np.array([0.])
    #     return [surface_map, surface_map, surface_map, surface_map]


    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
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
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_p = cell_sol_list[0]

            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p = cell_shape_grads_list[0]

            
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p = cell_v_grads_JxW_list[0]

            
            cell_JxW_p = cell_JxW[0]
            

            ######################
            ## Hu: Weak form of LHS
            ######################
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0]
            dt = self.params['dt']

            ## Hu: (num_quads,)
            tmp3 = p / dt
            ## Hu: (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val3 = np.sum(tmp3[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)


            weak_form = [val3] # [(num_nodes_p, vec_p)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel



# [RHS]
class AllenCahnForce(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable - order parameter eta
        self.fe_p = self.fes[0]

        self.params = params

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         # Some small noise to guide the dynamic relaxation solver
    #         return np.array([0.])
    #     return [surface_map, surface_map, surface_map, surface_map]


    def get_universal_kernel(self):
        ### Hu: AD can be directly used here (jax.grad)
        def f_local_grad(p):
            return 4*p*(p-1.0)*(p-0.5)

        vmap_f_local_grad = jax.jit(jax.vmap(f_local_grad))



        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
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
            
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_p = cell_sol_list[0]
            
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p = cell_shape_grads_list[0]

            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p = cell_v_grads_JxW_list[0]

            cell_JxW_p = cell_JxW[0]
            

            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)



            ######################
            ## Hu: Weak form of RHS
            ######################
            # Hu: Handle the term for gradient energy
            MnV = self.params['MnV']
            KnV = self.params['KnV']
            tmp1 = MnV * KnV * p_grads # (num_quads, vec_p, dim)
            val1 = np.sum(tmp1[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))


            # Hu: Handle the term for free energy
            tmp2 = MnV * vmap_f_local_grad(p) # (num_quads,)
            # tmp2 = MnV * vmap_f_local_grad(p_old) # (num_quads,)
            # print(tmp2.shape)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val2 = np.sum(tmp2[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)
            # print("val2", val2.shape)


            weak_form = [val1 + val2] # [(num_nodes_p, vec_p)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel



def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:06d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)



def get_mass(problem_mass):
    ## Hu: (fe.num_total_nodes) * (fe.vec) * (No. of fe)
    dofs = np.zeros(problem_mass.num_total_dofs_all_vars)
    
    sol_list = problem_mass.unflatten_fn_sol_list(dofs)
    problem_mass.newton_update(sol_list)

    ## Hu: Creating sparse matrix with scipy...
    A_sp_scipy = scipy.sparse.csr_array((onp.array(problem_mass.V), (problem_mass.I, problem_mass.J)),
        shape=(problem_mass.num_total_dofs_all_vars, problem_mass.num_total_dofs_all_vars))

    M = A_sp_scipy.sum(axis=1)

    return M



def get_explicit_dynamics(problem_mass, problem_force):
    M = get_mass(problem_mass)
    M_inv = 1./M

    def force_func(sol_list):
        internal_vars = []
        
        M_inv_list = problem_force.unflatten_fn_sol_list(M_inv)
        
        res_list = problem_force.compute_residual_vars(sol_list, internal_vars, problem_force.internal_vars_surfaces)
        
        ## Hu: rhs = -M_inv * res
        rhs_list = jax.tree.map(lambda x, y: -x * y, M_inv_list, res_list)
        return rhs_list

    @jax.jit
    def explicit_euler(sol_list):
        rhs_list = force_func(sol_list)

        sol_list = jax.tree.map(lambda x, y: x + y, sol_list, rhs_list)
        return sol_list

    return explicit_euler




def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)


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





    ### Hu: Variable: [p]
    problem_mass = AllenCahnMass(mesh, vec=1, dim=2, ele_type=ele_type, additional_info=[params])
    problem_force = AllenCahnForce(mesh, vec=1, dim=2, ele_type=ele_type, additional_info=[params])


    ## Hu: Definition of Neumann B.C.
    # def neumann_val(point):
    #     return np.array([0.])
    # location_fns = [left, right, top, bottom]
    # problem_mass = AllenCahnMass(mesh, vec=1, dim=2, ele_type=ele_type, location_fns=location_fns, additional_info=[params])
    # problem_force = AllenCahnForce(mesh, vec=1, dim=2, ele_type=ele_type, location_fns=location_fns, additional_info=[params])


    points = problem_mass.fe_p.points


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
    save_sols(problem_mass, sol_list, 0)
    

    explicit_euler = get_explicit_dynamics(problem_mass, problem_force)


    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
        sol_list = explicit_euler(sol_list)
        
        if (i + 1) % 1000 == 0:
            print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")

            save_sols(problem_mass, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is explicit solver for Allen-Cahn benchmark:", end_time - start_time)
