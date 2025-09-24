"""
Implicit finite element solver 2D for Cahn-Hilliard
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
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, box_mesh_gmsh, box_mesh, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem

from jax import config
config.update("jax_enable_x64", True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk/CH')
os.makedirs(vtk_dir, exist_ok=True)
 

class CahnHilliard(Problem):
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
        ### TODO: AD-based f_local
        ## f,c
        def dfdc_func(c):
            df = 4.*(c-1.)*(c-0.5)*c
            return df

        vmap_dfdc_func = jax.jit(jax.vmap(dfdc_func))

        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_sol_p_old, p_old, cell_sol_mu_old, mu_old):
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
            MnV = self.params['MnV']
            tmp2 = MnV * mu_grads
            
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
            # Handles the term `dfdc*v*dx` [Left hand side]
            # tmp4 = vmap_dfdc_func(p) # (num_quads,)
            tmp4 = vmap_dfdc_func(p_old) # (num_quads,)
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_mu, vec_mu)
            val4 = -np.sum(tmp4[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)


            # Handles the term `inner(grad(p), grad(v)*dx` [Mass Term]
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            KnV = self.params['KnV']
            tmp5 = KnV * p_grads # (num_quads, vec_p, dim)
            

            # (num_quads, 1, vec_mu, dim) * (num_quads, num_nodes_mu, 1, dim)
            # (num_quads, num_nodes_mu, vec_mu, dim) -> (num_nodes_mu, vec_mu)
            val5 = -np.sum(tmp5[:, None, :, :] * cell_v_grads_JxW_mu, axis=(0, -1))


            # [sol_p, sol_mu]
            weak_form = [val1 + val2, val3 + val4 + val5] # [(num_nodes_p, vec_p), (num_nodes_mu, vec_mu)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel


    def set_params(self, params):
        # Override base class method.
        sol_p_old, sol_mu_old = params

        ## [cell_sol_p_old, p_old, cell_sol_mu_old, mu_old]
        self.internal_vars = [sol_p_old[self.fe_p.cells],
                              self.fe_p.convert_from_dof_to_quad(sol_p_old)[:, :, 0], 
                              sol_mu_old[self.fe_mu.cells],
                              self.fe_mu.convert_from_dof_to_quad(sol_mu_old)[:, :, 0]]


def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'CH_p_{step:05d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)

    # vtk_path_mu = os.path.join(vtk_dir, f'CH_mu_{step:05d}.vtu')
    # save_sol(problem.fe_mu, sol_list[1], vtk_path_mu)


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

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
    

    ## Hu: Definition of Neumann B.C.
    # def neumann_val(point):
    #     return np.array([0.])

    # location_fns = [left, right, back, front]
    # problem = CahnHilliard(mesh, vec=1, dim=3, ele_type=ele_type, location_fns=location_fns, additional_info=[params])
    # problem = CahnHilliard(mesh, vec=1, dim=2, ele_type=ele_type, location_fns=location_fns, additional_info=[params])


    ### Hu: Positions of fe_p
    ## Hu: (4225, 2) - ((nx+1)*(nx+1), dim)
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

    save_sols(problem, sol_list, 0)
    

    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
        
        # [sol_p, sol_mu]
        problem.set_params([sol_list[0], sol_list[1]])


        sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list, 'tol': 1e-9})   
        
        
        if (i + 1) % 10 == 0:
            save_sols(problem, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is implicit solver for Cahn-Hilliard benchmark:", end_time - start_time)
