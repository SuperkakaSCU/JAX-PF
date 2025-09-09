"""
Explicit finite element solver for spinodal decomposition
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)



# [LHS]
class CahnHilliardMass(Problem):
    """If the problem structure is MU = F(U), this class handles the mass matrix M.
    """
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_p = self.fes[0]
        # ## Hu: mu variable for forth order PDE
        self.fe_mu = self.fes[1]

        self.WcV = 1.0   ## double well coefficient
        self.theta = 0.5
        self.params = params


    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_p, cell_sol_mu = cell_sol_list
            

            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_p, cell_shape_grads_mu = cell_shape_grads_list

            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_p, cell_v_grads_JxW_mu = cell_v_grads_JxW_list

            cell_JxW_p, cell_JxW_mu = cell_JxW[0], cell_JxW[1]
            


            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] 

            
            dt = self.params['dt']
            tmp1 = p / dt 
            val1 = np.sum(tmp1[:, None, None] * self.fe_p.shape_vals[:, :, None] * cell_JxW_p[:, None, None], axis=0)


            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            tmp2 = mu # (num_quads,)
            val2 = np.sum(tmp2[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)



            weak_form = [val1, val2] # [(num_nodes_p, vec_p), (num_nodes_mu, vec_mu)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return jax.jit(universal_kernel)
        # return universal_kernel



# [RHS]
class CahnHilliardForce(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_p = self.fes[0]
        # ## Hu: mu variable for forth order PDE
        self.fe_mu = self.fes[1]

        self.theta = 0.5
        self.WcV = 1.0   ## double well coefficient
        self.params = params



    def get_universal_kernel(self):
        ### TODO: AD-based f_local
        ## f,c
        def dfdc_func(c):
            # df = 4.*(c-1.)*(c-0.5)*c
            df = 1.0 * self.WcV *(c-1.)*(c-0.5)*c
            return df

        vmap_dfdc_func = jax.jit(jax.vmap(dfdc_func))


        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            
            cell_sol_p, cell_sol_mu = cell_sol_list
            

            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]

            cell_shape_grads_p, cell_shape_grads_mu = cell_shape_grads_list


            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            
            cell_v_grads_JxW_p, cell_v_grads_JxW_mu = cell_v_grads_JxW_list


            cell_JxW_p, cell_JxW_mu = cell_JxW[0], cell_JxW[1]
            


        
            p_grads = np.sum(cell_sol_p[None, :, :, None] * cell_shape_grads_p[:, :, None, :], axis=1) 
            p = np.sum(cell_sol_p[None, :, :] * self.fe_p.shape_vals[:, :, None], axis=1)[:, 0] 


            mu_grads = np.sum(cell_sol_mu[None, :, :, None] * cell_shape_grads_mu[:, :, None, :], axis=1) 
            mu = np.sum(cell_sol_mu[None, :, :] * self.fe_mu.shape_vals[:, :, None], axis=1)[:, 0]
            

            MnV = self.params['MnV']
            tmp2 = MnV * mu_grads
            val2 = np.sum(tmp2[:, None, :, :] * cell_v_grads_JxW_p, axis=(0, -1))
            
           
            tmp4 = vmap_dfdc_func(p) # (num_quads,)
            val4 = -np.sum(tmp4[:, None, None] * self.fe_mu.shape_vals[:, :, None] * cell_JxW_mu[:, None, None], axis=0)
            

            KnV = self.params['KnV']
            tmp5 = KnV * p_grads # (num_quads, vec_p, dim)
            val5 = -np.sum(tmp5[:, None, :, :] * cell_v_grads_JxW_mu, axis=(0, -1))
            

            weak_form = [val2, val4 + val5] # [(num_nodes_p, vec_p), (num_nodes_T, vec_T)]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return jax.jit(universal_kernel)



def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:06d}.vtu')
    save_sol(problem.fe_p, sol_list[0], vtk_path_p)
    



def get_mass(problem_mass):
    dofs = np.zeros(problem_mass.num_total_dofs_all_vars)
    print("dofs", dofs.shape)

    sol_list = problem_mass.unflatten_fn_sol_list(dofs)
    

    problem_mass.newton_update(sol_list)

    
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
        

        rhs_list = jax.tree.map(lambda x, y: -x * y, M_inv_list, res_list)
        return rhs_list

    @jax.jit
    def explicit_euler(sol_list):
        
        rhs_list = force_func(sol_list)
        rhs_p, rhs_mu = rhs_list

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

    json_file = os.path.join(input_dir, 'json/params_CH_2D.json')
    params = json_parse(json_file)


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


    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    # Lz = np.max(mesh.points[:, 2])
    print("Lx: {0}, Ly:{1}".format(Lx, Ly))
    

    

    def left(point):
        return np.isclose(point[0], 0., atol=1e-3)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-3)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-3)

    def back(point):
        return np.isclose(point[1], Ly, atol=1e-3)

    

    def neumann_val(point):
        return np.array([0.])


    location_fns = [left, right, back, front]



    problem_mass = CahnHilliardMass(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type],  additional_info=[params])
    problem_force = CahnHilliardForce(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type],  additional_info=[params])


    
    points = problem_mass.fe_p.points
    print("points", points.shape)


    # The average composition
    c0 = 0.50

    # The initial perturbation amplitude
    icamplitude = 0.01

    
    sol_p = np.ones((problem_mass.fe_p.num_total_nodes, problem_mass.fe_p.vec)) * c0

    chi = jax.random.uniform(jax.random.PRNGKey(0), shape=(problem_mass.fe_p.num_total_nodes, problem_mass.fe_p.vec))

    sol_p = sol_p + (chi - 0.5) * icamplitude * 2.0
    
    sol_mu = np.zeros((problem_mass.fe_p.num_total_nodes, problem_mass.fe_p.vec))

    sol_list = [sol_p, sol_mu]
    save_sols(problem_mass, sol_list, 0)
    
    

    explicit_euler = get_explicit_dynamics(problem_mass, problem_force)

    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
    
        sol_list = explicit_euler(sol_list)
        # sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list})   
        # sol_list = solver(problem, solver_options={'petsc_solver': {}, 'initial_guess': sol_list})   
        # sol_list = solver(problem, solver_options={'umfpack': {}, 'initial_guess': sol_list})   

        if (i + 1) % 1000 == 0:
            print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
            save_sols(problem_mass, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is explicit solver for spinodal decomposition:", end_time - start_time)
