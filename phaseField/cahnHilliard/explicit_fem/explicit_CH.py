"""
Explicit finite element solver for 2D Cahn Hilliard
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)



# [LHS]
class CahnHilliardMass(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_p = self.fes[0]
        ## Hu: mu variable for forth order PDE
        self.fe_mu = self.fes[1]

        self.theta = 0.5
        self.params = params

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         # Some small noise to guide the dynamic relaxation solver
    #         return np.array([0.])
    #     return [surface_map, surface_map, surface_map, surface_map]


    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            """
            Handles the weak form with one cell.
            Assume trial function (p, mu), test function (q, w)

            cell_sol_flat: (num_nodes*vec + ...,)
            cell_sol_list: [(num_nodes, vec), ...]
            x: (num_quads, dim)  ->  physical_quad_points
            cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_JxW: (num_vars, num_quads)
            cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            """
            ##### p is phase field variable of 'concentration'

            #### Hu: Unassemble the values to different variables
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
            


            ######################
            ## Hu: This is phase field variable weak form of p on each cell (Test function: q)
            ######################
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            ##################### (num_nodes_p, vec_p)
            dt = self.params['dt']
            tmp1 = p / dt # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val1 = np.sum(tmp1[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)


            ######################
            ## This is phase field variable weak form of mu on each cell (Test function: w)
            ######################
            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            tmp2 = mu # (num_quads,)
            # (num_quads, 1, vec_mu) * (num_quads, num_nodes_mu, 1) * (num_quads, 1, 1) -> (num_nodes_mu, vec_mu)
            val2 = np.sum(tmp2[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)


            weak_form = [val1, val2] # [(num_nodes_p, vec_p), (num_nodes_mu, vec_mu)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel



# [RHS]
class CahnHilliardForce(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_p = self.fes[0]
        # ## Hu: mu variable for forth order PDE
        self.fe_mu = self.fes[1]

        self.theta = 0.5

        self.params = params

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         # Some small noise to guide the dynamic relaxation solver
    #         return np.array([0.])
    #     return [surface_map, surface_map, surface_map, surface_map]


    def get_universal_kernel(self):
        ### Hu: AD can be directly used here (jax.grad)
        def dfdc_func(c):
            df = 4.*(c-1.)*(c-0.5)*c
            return df

        vmap_dfdc_func = jax.jit(jax.vmap(dfdc_func))


        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            """
            Handles the weak form with one cell.
            Assume trial function (p, mu), test function (q, w)

            cell_sol_flat: (num_nodes*vec + ...,)
            cell_sol_list: [(num_nodes, vec), ...]
            x: (num_quads, dim)  ->  physical_quad_points
            cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_JxW: (num_vars, num_quads)
            cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            """


            #### Hu: Unassemble the values to different variables
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
            

            ## Value of c and mu
            # Handles the term `inner(..., grad(q)*dx` [Hybrid implicit/explicit]
            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)


            mu_grads = np.sum(cell_sol_mu[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)
            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            


            ######################
            ## Hu: This is phase field variable weak form of p on each cell (Test function: q)
            ######################
            ## Hu: Handles the term `inner(grad(mu), grad(w)*dx` [Mass Term]
            MnV = self.params['MnV']
            tmp2 = MnV * mu_grads
            # (num_quads, 1, vec_p, dim) * (num_quads, num_nodes_p, 1, dim) -> (num_nodes_p, vec_p)
            val2 = np.sum(tmp2[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))


            ######################
            ## This is phase field variable weak form of mu on each cell (Test function: w)
            ######################
            ## Hu: Handles the term `dfdc*w*dx` [Left hand side]
            tmp4 = vmap_dfdc_func(p) # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val4 = -np.sum(tmp4[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)


            ## Hu: Handles the term `inner(grad(p), grad(w)*dx` [Mass Term]
            ## Hu: (1,) * (num_quads, vec_mu, dim) -> (num_quads, vec_mu, dim)
            KnV = self.params['KnV']
            tmp5 = KnV * p_grads # (num_quads, vec_p, dim)
            # (num_quads, 1, vec_mu, dim) * (num_quads, num_nodes_mu, 1, dim) -> (num_nodes_mu, vec_mu)
            val5 = -np.sum(tmp5[:, None, :, :] * cell_v_grads_JxW_mu, axis=(0, -1))


            # [sol_p, sol_mu]
            weak_form = [val2, val4 + val5] # [(num_nodes_p, vec_p), (num_nodes_T, vec_T)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel



def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:06d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)



def get_mass(problem_mass):
    ## Hu: (fe.num_total_nodes) * (fe.vec) * (No. of fe)
    dofs = np.zeros(problem_mass.num_total_dofs_all_vars)

    ## Hu: [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
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

        ## Hu: Computing cell residual
        res_list = problem_force.compute_residual_vars(sol_list, internal_vars, problem_force.internal_vars_surfaces)
        
        ## Hu: rhs = -M_inv * res
        rhs_list = jax.tree.map(lambda x, y: -x * y, M_inv_list, res_list)
        return rhs_list

    @jax.jit
    def explicit_euler(sol_list):
        rhs_list = force_func(sol_list)
        rhs_p, rhs_mu = rhs_list

        ## Hu: sol = sol + rhs
        sol_p, sol_mu = sol_list

        sol_p = sol_p + rhs_p
        sol_mu = rhs_mu

        sol_list = [sol_p, sol_mu]
        return sol_list

    return explicit_euler



def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params_CH.json')
    params = json_parse(json_file)
    # print("params", params)

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
    problem_mass = CahnHilliardMass(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type],  additional_info=[params])
    problem_force = CahnHilliardForce(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type],  additional_info=[params])


    ## Hu: Definition of Neumann B.C.
    # def neumann_val(point):
    #     return np.array([0.])

    # location_fns = [left, right, back, front]
    # problem_mass = CahnHilliardMass(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type], location_fns=[location_fns, location_fns], additional_info=[params])
    # problem_force = CahnHilliardForce(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type], location_fns=[location_fns, location_fns], additional_info=[params])

    
    ### Hu: Positions of fe_p
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
    sol_mu = np.zeros((problem_mass.fe_p.num_total_nodes, problem_mass.fe_p.vec))
    sol_list = [sol_p, sol_mu]

    save_sols(problem_mass, sol_list, 0)
    
    explicit_euler = get_explicit_dynamics(problem_mass, problem_force)


    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
        sol_list = explicit_euler(sol_list)

        if (i + 1) % 10000 == 0:
            print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
            save_sols(problem_mass, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is explicit solver for Cahn-Hilliard benchmark:", end_time - start_time)
