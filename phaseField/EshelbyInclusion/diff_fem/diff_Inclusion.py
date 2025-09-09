"""
Diff. based on AD
Implicit finite element solver for Eshelby inclusion
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob

from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem
from jax_fem import logger

from jax_fem.solver import implicit_vjp


from jax import config
config.update("jax_enable_x64", True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
# vtk_dir = os.path.join(output_dir, 'vtk')
# os.makedirs(vtk_dir, exist_ok=True)


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


class Inclusion(Problem):
    def custom_init(self, params):
        ## Hu: displacement u
        self.fe_u = self.fes[0]

        ## Hu: elastic modulus
        self.C = onp.zeros((self.dim, self.dim, self.dim, self.dim))

        ## 3D definition
        # E = 22.5
        # nu = 0.3
        # C11 = E*(1-nu)/((1+nu)*(1-2*nu))
        # C12 = E*nu/((1+nu)*(1-2*nu))
        # C44 = E/(2.*(1. + nu))

        # self.C[0, 0, 0, 0] = C11
        # self.C[1, 1, 1, 1] = C11
        # self.C[2, 2, 2, 2] = C11

        # self.C[0, 0, 1, 1] = C12
        # self.C[1, 1, 0, 0] = C12

        # self.C[0, 0, 2, 2] = C12
        # self.C[2, 2, 0, 0] = C12

        # self.C[1, 1, 2, 2] = C12
        # self.C[2, 2, 1, 1] = C12

        # self.C[1, 2, 1, 2] = C44
        # self.C[1, 2, 2, 1] = C44
        # self.C[2, 1, 1, 2] = C44
        # self.C[2, 1, 2, 1] = C44

        # self.C[2, 0, 2, 0] = C44
        # self.C[2, 0, 0, 2] = C44
        # self.C[0, 2, 2, 0] = C44
        # self.C[0, 2, 0, 2] = C44

        # self.C[0, 1, 0, 1] = C44
        # self.C[0, 1, 1, 0] = C44
        # self.C[1, 0, 0, 1] = C44
        # self.C[1, 0, 1, 0] = C44


        E = 22.5
        nu = 0.3

        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))

        self.C[0, 0, 0, 0] = self.lam + 2 * self.mu
        self.C[0, 0, 1, 1] = self.lam
        self.C[1, 1, 0, 0] = self.lam
        self.C[1, 1, 1, 1] = self.lam + 2 * self.mu

        self.C[0, 1, 0, 1] = self.mu
        self.C[0, 1, 1, 0] = self.mu
        self.C[1, 0, 0, 1] = self.mu
        self.C[1, 0, 1, 0] = self.mu

        self.params = params

        lam_gp = self.lam*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))
        mu_gp = self.mu*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        self.internal_vars = [lam_gp, mu_gp]

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0.])
        return [surface_map, surface_map]


    ## Hu: The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    ## solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    ## Hu: Work on each quad point
    def get_tensor_map(self):
        ## Hu: (u_grads, *internal_vars)
        def stress(u_grad, sol_u_old, eps0_diag, lam, mu):
            epsilon0 = np.eye(self.fe_u.dim)*eps0_diag
            
            epsilon = 0.5 * (u_grad + u_grad.T)

            epsilon_ctr = epsilon - epsilon0

            sigma = lam * np.trace(epsilon_ctr) * np.eye(self.dim) + 2*mu*epsilon_ctr

            return sigma
        return stress



    def set_initial_params(self, initial_params):
        # Override base class method.
        sol_u_old, eps0_diag = initial_params
        eps0_diag = eps0_diag.reshape(-1, 1)

        # (num_total_nodes, vec) -> (num_cells, num_nodes, vec)
        cells_eps0 = eps0_diag[self.fe_u.cells]
        
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        eps0_diag_quad = np.sum(cells_eps0[:, None, :, :] * self.fe_u.shape_vals[None, :, :, None], axis=2)

        self.initial_internal_vars = [sol_u_old[self.fe_u.cells],
                                      eps0_diag_quad]

        self.internal_vars = self.initial_internal_vars + self.internal_vars

    def set_params(self, params):
        self.internal_vars = params

    def set_initial_guess(self, initial_sol):
        self.initial_guess = initial_sol


    def crt_volume(self, sol):
        def det_fn(u_grad):
            F = u_grad + np.eye(self.dim)
            return np.linalg.det(F)

        u_grads = self.fes[0].sol_to_grad(sol)
        vmap_det_fn = jax.jit(jax.vmap(jax.vmap(det_fn)))
        crt_volume = np.sum(vmap_det_fn(u_grads) * self.fes[0].JxW)

        return crt_volume



def save_sols(problem, sol_list, step):
    vtk_path_u = os.path.join(vtk_dir, f'u_{step:06d}.vtu')
    save_sol(problem.fe_u, sol_list[0], vtk_path_u)


def problem():
    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    

    Lx = params['Lx']
    Ly = params['Ly']
    Lz = params['Lz']
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']

    ## Hu: 2D problem
    ele_type = 'QUAD4'  # polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)

    ## Hu: 3D problem
    # ele_type = 'HEX8'  # polynomial of the element: 1
    # cell_type = get_meshio_cell_type(ele_type)
    # meshio_mesh = box_mesh(Nx=nx, Ny=ny, Nz=nz, domain_x=Lx, domain_y=Ly, domain_z=Lz)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    ## Hu: Sizes of domain
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    print("Lx:{0}, Ly:{0}".format(Lx, Ly))


    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)



    ### Hu: Define Neumann B.C.
    location_fns = [left, bottom]
    def zero_dirichlet_val(point):
        return 0.

    
    dirichlet_bc_info = [[top, top, right, right],
                         [0, 1, 0, 1],
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]


    ### Hu: [u]
    problem = Inclusion(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns, additional_info=[params])

    

    ### Hu: Positions of fe_u
    points = problem.fe_u.points


    ## Hu: Definition of initial condition
    core_center = np.array([0., 0.])
    r = np.linalg.norm(points - core_center, axis=1)


    a = 10.0     # radius of inclusion
    l = 20.0     # sharpness of interface
    m = 0.01     # misfit amplitude

    eps0_diag = 0.01 * (0.5 + 0.5 * (1.0 - np.exp(-20.0 * (r - a))) / 
                                (1.0 + np.exp(-20.0 * (r - a))))

    sfts = np.zeros((len(r), problem.dim, problem.dim))


    ## Hu: (1, None, None) * (self.dim, self.dim) ->  (num_dofs, self.dim, self.dim)
    sfts = eps0_diag[:, None, None] * np.eye(problem.dim) # broadcasting to (num_quads, 3, 3)
    
    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]

    initial_params = [sol_list[0], eps0_diag]
    
    problem.set_initial_params(initial_params)
    problem.set_initial_guess(sol_list)

    fwd_pred = ad_wrapper(problem)

    def simulation(alpha):
        params = problem.internal_vars
        print("params", len(params))


        coeff1, coeff2 = alpha

        # lam
        params[2] = coeff1*params[2]
        # mu
        params[3] = coeff2*params[3]


        print("Start Solving PDE ...")
        sol_list = fwd_pred(params)       

        obj_val = problem.crt_volume(sol_list[0])

        jax.debug.print("obj_val={x}", x=obj_val)

        return obj_val


    # AD grads [12.447285 -6.223642]
    # FDM: [12.447378734759695, -6.223727295582648]

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
    print("AD Testing")
    problem()

    
