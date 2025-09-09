"""
Implicit finite element solver for Allen Cahn
Application: grain growth
S.P. Gentry and K. Thornton, IOP Conf. Series: Materials Science and Engineering 89 (2015) 012024.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)



class AllenCahn(Problem):
    def custom_init(self, params):
        self.params = params

        self.num_ops = len(self.fes)  # Number of order parameters

        ## Hu: list of order parameters
        self.fe_ns = self.fes



    def get_universal_kernel(self):
        ## n - (num_ops, num_quads); n_sum - (num_quads,)
        def f_local_grad(n, n_sum):
            alpha = self.params['alpha']
            f_local_grad = -n + n**3.0
            f_inter_grad = 2 * alpha * n * (n_sum - n*n)

            return f_local_grad + f_inter_grad

        vmap_f_local_grad = jax.jit(jax.vmap(f_local_grad, in_axes=(0, None)))


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
            
            num_vars = self.num_vars
            vec = self.fe_ns[0].vec
            num_nodes = self.fe_ns[0].num_nodes
            num_quads = self.fe_ns[0].num_quads
            dim = self.fe_ns[0].dim
            num_ops = self.num_ops

            

            #### Hu: Unassemble the values to different variables
            cell_sol_old_array = np.stack(cell_internal_vars[:num_ops], axis=0)  # (num_vars, num_nodes, vec)
            quad_old_array = np.stack(cell_internal_vars[num_ops:], axis=0)      # (num_vars, num_quads)


            ## Hu: [n0, n1, n2, n3, n4, n5] - [(num_nodes, vec), (num_nodes, vec), ...]
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            # cell_sol_n0, cell_sol_n1, cell_sol_n2, cell_sol_n3, cell_sol_n4, cell_sol_n5 = cell_sol_list
            cell_sol_array = np.stack(cell_sol_list, axis=0)     # (num_vars, num_nodes, vec)
            
            
            ## Hu: [(num_quads, num_nodes, dim), ...]
            # cell_shape_grads_n0, cell_shape_grads_n1, cell_shape_grads_n2, cell_shape_grads_n3, cell_shape_grads_n4, cell_shape_grads_n5 = cell_shape_grads_list
            shape_grads_list = np.stack([cell_shape_grads[:, self.num_nodes_cumsum[i]:self.num_nodes_cumsum[i+1], :] for i in range(num_ops)], axis=0)

            ## Hu: [(num_quads, num_nodes, 1, dim), ...]
            # cell_v_grads_JxW_n0, cell_v_grads_JxW_n1, cell_v_grads_JxW_n2, cell_v_grads_JxW_n3, cell_v_grads_JxW_n4, cell_v_grads_JxW_n5 = cell_v_grads_JxW_list
            v_grads_JxW_list = np.stack([cell_v_grads_JxW[:, self.num_nodes_cumsum[i]:self.num_nodes_cumsum[i+1], :, :] for i in range(num_ops)], axis=0)

            ## cell_JxW: [(num_quads,), ...]
            # cell_JxW_n0 = cell_JxW[0]
            JxW_list = np.stack([cell_JxW[i] for i in range(num_ops)], axis=0)


            ######################
            ## This is phase field variable weak form
            ######################
            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)

            ## Hu: [(num_quads, vec_p, dim), ...]
            n_grads = jax.vmap(lambda cell_sol_n0, cell_shape_grads_n0: np.sum(cell_sol_n0[None, :, :, None] * cell_shape_grads_n0[:, :, None, :], axis=1), in_axes=(0,0))(cell_sol_array, shape_grads_list)
            

            ## Hu: (num_ops, num_quads, num_nodes)
            shape_vals_list = np.stack([fe.shape_vals[:, :] for fe in self.fe_ns], axis=0)
            ## Hu: (num_ops, num_quads)
            n_vals = jax.vmap(lambda cell_sol_n0, shape_vals: np.sum(cell_sol_n0[None, :, :] * shape_vals[:, :, None], axis=1)[:, 0])(cell_sol_array, shape_vals_list)
            
            ## Hu: n_sum: (num_quads,)
            n_sum = np.sum(n_vals**2, axis=0)

            
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            MnV = self.params['MnV'] # L in Eqs
            KnV = self.params['KnV']


            grad_term = MnV * KnV * n_grads  # (num_vars, num_quads, vec, dim)
            val1 = np.sum(grad_term[:, :, None, :, :] * v_grads_JxW_list, axis=(1, -1))  # (num_vars, num_nodes, vec)
            # print("grad_term", grad_term.shape)
            # print("val1", val1.shape)

            

            ##################### (num_nodes_p, vec_p)
            chem = MnV * vmap_f_local_grad(n_vals, n_sum)  # (num_vars, num_quads)
            # (num_vars, num_nodes, vec)
            val2 = jax.vmap(lambda chem_0, shape_vals, JxW_0: np.sum(chem_0[:, None, None] * shape_vals[:, :, None] * JxW_0[:, None, None], axis=0))(chem, shape_vals_list, JxW_list)
            # print("chem", chem.shape)
            # print("val2", val2.shape)
            
            
            dt = self.params['dt']
            dndt = (n_vals - quad_old_array) / dt # (num_vars, num_quads)
            # (num_vars, num_nodes, vec)
            val3 = jax.vmap(lambda dndt_0, shape_vals, JxW_0: np.sum(dndt_0[:, None, None] * shape_vals[:, :, None] * JxW_0[:, None, None], axis=0))(dndt, shape_vals_list, JxW_list)
            # print("dndt", dndt.shape)
            # print("val3", val3.shape)
            


            val = val1 + val2 + val3


            weak_form = [val[i] for i in range(val.shape[0])]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel

    def set_params(self, sol_list_old):
        self.internal_vars = []

        for i, fe in enumerate(self.fe_ns):
            self.internal_vars.append(sol_list_old[i][fe.cells])                              # dof-level (num_cells, num_nodes, vec)
        for i, fe in enumerate(self.fe_ns):
            self.internal_vars.append(fe.convert_from_dof_to_quad(sol_list_old[i])[:, :, 0])  # quad-level (num_cells, num_quads)

        


def compute_max_op_from_sol_list_jax(sol_list, threshold=0.01):
    sol_list = [np.squeeze(s) for s in sol_list]

    op_array = np.stack(sol_list, axis=1)

    max_op_idx = np.argmax(op_array, axis=1)
    op_sum = np.sum(op_array, axis=1)
    max_op_idx = np.where(op_sum < threshold, -1, max_op_idx)

    return max_op_idx


def save_sols(problem, sol_list, step):
    vtk_path = os.path.join(vtk_dir, f'p0_{step:06d}.vtu')
    max_op_idx = compute_max_op_from_sol_list_jax(sol_list, threshold=0.01)
    infos = [(f'n{i}', sol_list[i]) for i in range(len(sol_list))]
    infos.append(('obj_id', max_op_idx))
    save_sol(problem.fe_ns[0], sol_list[0], vtk_path, point_infos=infos)


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
    Lz = params['Lz']
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']
    ele_type = 'QUAD4'  # polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    ## Hu: Sizes of domain
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    print("Lx:{0}, Ly:{1}".format(Lx, Ly))

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def top(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], Ly, atol=1e-5)


    ## Hu: number of order parameters
    num_ops = 15

    problem = AllenCahn([mesh] * num_ops, vec=[1] * num_ops, dim=2, ele_type=[ele_type] * num_ops, \
        additional_info=[params])


    points = problem.fe_ns[0].points
    

    ## Hu: Definition of initial condition
    def set_ICs(sol_IC, p, domain_size, index, center, rad):
        p = p.reshape(nx+1, ny+1, problem.fes[0].dim)

        ## Hu: physical coordinte
        scaled_centers = center * domain_size
        dist = np.linalg.norm(p - scaled_centers, axis=2)
        
        rad = rad*domain_size[0]
        sol_IC = sol_IC.at[index].set(0.5 * (1.0 - np.tanh((dist - rad) / 0.5)))
       
        return sol_IC

    ## Work on each nucleation
    vmap_set_ICs = jax.jit(jax.vmap(set_ICs, in_axes=(None, None, None, 0, 0, 0)))

    ## Initialization of 6 order params (0~5)
    # (num_vars, nx, ny)
    sol_IC = np.array([np.zeros((nx+1, ny+1))] * num_ops)


    index_list  = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14])


    center_list = np.array([
            [0.2, 0.15], [0.25, 0.7], [0.5, 0.5], [0.6, 0.85], [0.85, 0.35],   # big
            [0.08, 0.92], [0.75, 0.6], [0.75, 0.1], [0.2, 0.45], [0.85, 0.85], # medium
            [0.55, 0.05], [0.1, 0.35], [0.95, 0.65], [0.9, 0.15], [0.45, 0.25] # small
            ])

    # (num_nucli, )
    rad_list = np.array([0.14]*5 + [0.08]*5 + [0.05]*5)
    domain_size = np.array([Lx, Ly])

    # (num_nucli, num_vars, nx+1, nx+1)
    sol_IC_nucl = vmap_set_ICs(sol_IC, points, domain_size, index_list, center_list, rad_list)
    # print("sol_IC_nucl", sol_IC_nucl.shape)

    # (num_vars, nx+1, nx+1)    
    sol_IC_nucl = np.sum(sol_IC_nucl, axis=0)
    sol_IC_nucl = np.minimum(sol_IC_nucl, 0.999)
    sol_IC_nucl = sol_IC_nucl.reshape(problem.num_vars, -1)

    
    sol_list = [sol_IC_nucl[i].reshape(problem.fe_ns[i].num_total_nodes, problem.fe_ns[i].vec) for i in range(num_ops)]
    

    save_sols(problem, sol_list, 0)


    nIter = int(t_OFF/dt)
    print("nIter", nIter)

    for i in range(nIter + 1):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
        
        problem.set_params(sol_list)

        
        sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list, 'tol': 1e-7})   
        # sol_list = solver(problem, solver_options={'petsc_solver': {}, 'initial_guess': sol_list})   
        # sol_list = solver(problem, solver_options={'umfpack': {}, 'initial_guess': sol_list})   

        if (i + 1) % 5 == 0:
            save_sols(problem, sol_list, i + 1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is implicit solver for grain growth:", end_time - start_time)
