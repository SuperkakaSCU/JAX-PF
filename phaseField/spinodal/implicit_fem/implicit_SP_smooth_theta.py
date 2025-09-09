######### Note: Implicit doesn't reach convergence without theta-method

"""
Implicit finite element solver for Cahn-Hilliard
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        # ## Hu: mu variable for forth order PDE
        self.fe_mu = self.fes[1]

        self.theta = 0.5

        self.WcV = 1.0   ## double well coefficient
        self.theta = 0.5

        self.params = params

    

    def get_universal_kernel(self):
        ### TODO: AD-based f_local
        ## f,c
        def dfdc_func(c):
            # df = 4.*(c-1.)*(c-0.5)*c
            df = 1.0 * self.WcV *(c-1.)*(c-0.5)*c
            return df

        vmap_dfdc_func = jax.jit(jax.vmap(dfdc_func))

        
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_sol_p_old, p_old, cell_sol_mu_old, mu_old):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
        
            cell_sol_p, cell_sol_mu = cell_sol_list
            

            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            
            cell_shape_grads_p, cell_shape_grads_mu = cell_shape_grads_list


            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            
            cell_v_grads_JxW_p, cell_v_grads_JxW_mu = cell_v_grads_JxW_list

            
            cell_JxW_p, cell_JxW_mu = cell_JxW[0], cell_JxW[1]
            


            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            p_grads_old = np.sum(cell_sol_p_old[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) # (num_quads, vec_p, dim)

            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            p_old = np.sum(cell_sol_p_old[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)


            mu_grads = np.sum(cell_sol_mu[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)
            mu_grads_old = np.sum(cell_sol_mu_old[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) # (num_quads, vec_mu, dim)

            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            mu_old = np.sum(cell_sol_mu_old[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            


            dt = self.params['dt']
            tmp1 = (p - p_old)/dt # (num_quads,)
            val1 = np.sum(tmp1[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)
            


            MnV = self.params['MnV']
            cell_sol_mu_theta = (1.-self.theta)*cell_sol_mu_old + self.theta*cell_sol_mu  
            mu_theta_grads = np.sum(cell_sol_mu_theta[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) 
            tmp2 = MnV * mu_theta_grads 
            val2 = np.sum(tmp2[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))


            tmp3 = mu # (num_quads,)
            val3 = np.sum(tmp3[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)
            

            tmp4 = vmap_dfdc_func(p) # (num_quads,)
            val4 = -np.sum(tmp4[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)
            

            KnV = self.params['KnV']
            tmp5 = KnV * p_grads # (num_quads, vec_p, dim)
            
            
            val5 = -np.sum(tmp5[:, None, :, :] * cell_v_grads_JxW_mu, axis=(0, -1))
            

            # [sol_p, sol_mu]
            weak_form = [val1 + val2, val3 + val4 + val5] # [(num_nodes_p, vec_p), (num_nodes_T, vec_T)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return jax.jit(universal_kernel)
        # return universal_kernel

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
    vtk_path_u = os.path.join(vtk_dir, f'CH_mu_{step:05d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)
    save_sol(problem.fe_mu, sol_list[1], vtk_path_u)


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params_CH_2D.json')
    params = json_parse(json_file)
    # print("params", params)

    dt = params['dt']
    t_OFF = params['t_OFF']
    hx = params['hx']
    hy = params['hy']
    hz = params['hz']
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']

    # ele_type = 'TET10'  # polynomial of the element: 2
    ele_type = 'QUAD4'  # polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=nx*hx, domain_y=ny*hy)
    # meshio_mesh = box_mesh_gmsh(Nx=nx, Ny=ny, Nz=nz, Lx=nx*hx, Ly=ny*hy, Lz=nz*hz, data_dir=input_dir, ele_type=ele_type)
    # meshio_mesh = box_mesh(Nx=nx, Ny=ny, Nz=nz, domain_x=nx*hx, domain_y=ny*hy, domain_z=nz*hz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    ## Hu: Sizes of domain - (100, 100)
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    # Lz = np.max(mesh.points[:, 2])
    print("Lx: {0}, Ly:{1}".format(Lx, Ly))
    # print("Lx: {0}, Ly:{1}, Lz:{2}".format(Lx, Ly, Lz))

    

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


    def neumann_val(point):
        return np.array([0.])


    location_fns = [left, right, back, front]

    
    
    ### Hu: [p, mu]
    problem = CahnHilliard(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type], \
                             additional_info=[params])


    points = problem.fe_p.points
    
    # The average composition
    c0 = 0.50

    # The initial perturbation amplitude
    icamplitude = 0.01

    
    sol_p = np.ones((problem.fe_p.num_total_nodes, problem.fe_p.vec)) * c0

    chi = jax.random.uniform(jax.random.PRNGKey(0), shape=(problem.fe_p.num_total_nodes, problem.fe_p.vec))
    
    sol_p = sol_p + (chi - 0.5) * icamplitude * 2.0

    sol_mu = np.zeros((problem.fe_p.num_total_nodes, problem.fe_p.vec))
    

    sol_list = [sol_p, sol_mu]
    save_sols(problem, sol_list, 0)
    

    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
        
        # [sol_p, sol_mu]
        problem.set_params([sol_list[0], sol_list[1]])

        sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list, 'line_search_flag': False})   

        
        if (i + 1) % 1000 == 0:
            save_sols(problem, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is implicit solver for spinodal decomposition:", end_time - start_time)
