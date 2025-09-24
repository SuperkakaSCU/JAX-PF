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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)



class AllenCahn(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable from 0 ~ 5
        self.fe_n0 = self.fes[0]
        self.fe_n1 = self.fes[1]
        self.fe_n2 = self.fes[2]
        self.fe_n3 = self.fes[3]
        self.fe_n4 = self.fes[4]
        self.fe_n5 = self.fes[5]

        self.params = params

    # def get_surface_maps(self):
    #     def surface_map(u, x):
    #         # Some small noise to guide the dynamic relaxation solver
    #         return np.array([0.])
    #     return [surface_map, surface_map, surface_map, surface_map]


    def get_universal_kernel(self):
        ### Hu: AD can be directly used here (jax.grad)
        def f_local_grad(n, n_sum):
            alpha = self.params['alpha']
            f_local_grad = -n + n**3.0
            f_inter_grad = 2 * alpha * n * (n_sum - n*n)

            return f_local_grad + f_inter_grad

        vmap_f_local_grad = jax.jit(jax.vmap(f_local_grad, in_axes=(0, 0)))


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
            cell_sol_n0_old,  cell_sol_n1_old, cell_sol_n2_old, cell_sol_n3_old, cell_sol_n4_old, cell_sol_n5_old, \
            n0_old, n1_old, n2_old, n3_old, n4_old, n5_old = cell_internal_vars

            ## Hu: [n0, n1, n2, n3, n4, n5] - [(num_nodes, vec), (num_nodes, vec), ...]
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            ## (num_nodes_p, vec_p)
            cell_sol_n0, cell_sol_n1, cell_sol_n2, cell_sol_n3, cell_sol_n4, cell_sol_n5 = cell_sol_list

            ## Hu: cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_n0, cell_shape_grads_n1, cell_shape_grads_n2, cell_shape_grads_n3, cell_shape_grads_n4, cell_shape_grads_n5 = cell_shape_grads_list

            ## Hu: cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_n0, cell_v_grads_JxW_n1, cell_v_grads_JxW_n2, cell_v_grads_JxW_n3, cell_v_grads_JxW_n4, cell_v_grads_JxW_n5 = cell_v_grads_JxW_list

            ## cell_JxW: (num_vars, num_quads)
            cell_JxW_n0 = cell_JxW[0]
            cell_JxW_n1 = cell_JxW[1]
            cell_JxW_n2 = cell_JxW[2]
            cell_JxW_n3 = cell_JxW[3]
            cell_JxW_n4 = cell_JxW[4]
            cell_JxW_n5 = cell_JxW[5]

            

            ######################
            ## This is phase field variable weak form
            ######################
            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            n0_grads = np.sum(cell_sol_n0[None, :, :, None] * cell_shape_grads_n0[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n1_grads = np.sum(cell_sol_n1[None, :, :, None] * cell_shape_grads_n1[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n2_grads = np.sum(cell_sol_n2[None, :, :, None] * cell_shape_grads_n2[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n3_grads = np.sum(cell_sol_n3[None, :, :, None] * cell_shape_grads_n3[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n4_grads = np.sum(cell_sol_n4[None, :, :, None] * cell_shape_grads_n4[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n5_grads = np.sum(cell_sol_n5[None, :, :, None] * cell_shape_grads_n5[:, :, None, :], axis=1) # (num_quads, vec_p, dim)

            ## Hu: TODO: old grad terms are not needed
            n0_grads_old = np.sum(cell_sol_n0_old[None, :, :, None] * cell_shape_grads_n0[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n1_grads_old = np.sum(cell_sol_n1_old[None, :, :, None] * cell_shape_grads_n1[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n2_grads_old = np.sum(cell_sol_n2_old[None, :, :, None] * cell_shape_grads_n2[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n3_grads_old = np.sum(cell_sol_n3_old[None, :, :, None] * cell_shape_grads_n3[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n4_grads_old = np.sum(cell_sol_n4_old[None, :, :, None] * cell_shape_grads_n4[:, :, None, :], axis=1) # (num_quads, vec_p, dim)
            n5_grads_old = np.sum(cell_sol_n5_old[None, :, :, None] * cell_shape_grads_n5[:, :, None, :], axis=1) # (num_quads, vec_p, dim)   

            
            n0 = np.sum(cell_sol_n0[None, :, :] * self.fe_n0.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n1 = np.sum(cell_sol_n1[None, :, :] * self.fe_n1.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n2 = np.sum(cell_sol_n2[None, :, :] * self.fe_n2.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n3 = np.sum(cell_sol_n3[None, :, :] * self.fe_n3.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n4 = np.sum(cell_sol_n4[None, :, :] * self.fe_n4.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n5 = np.sum(cell_sol_n5[None, :, :] * self.fe_n5.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            n_sum = n0**2.0 + n1**2.0 + n2**2.0 + n3**2.0 + n4**2.0 + n5**2.0  # (num_quads,)

            
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            MnV = self.params['MnV'] # L in Eqs
            KnV = self.params['KnV']
            tmp1_0 = MnV * KnV * n0_grads # (num_quads, vec_p, dim)
            tmp1_1 = MnV * KnV * n1_grads # (num_quads, vec_p, dim)
            tmp1_2 = MnV * KnV * n2_grads # (num_quads, vec_p, dim)
            tmp1_3 = MnV * KnV * n3_grads # (num_quads, vec_p, dim)
            tmp1_4 = MnV * KnV * n4_grads # (num_quads, vec_p, dim)
            tmp1_5 = MnV * KnV * n5_grads # (num_quads, vec_p, dim)

            val1_0 = np.sum(tmp1_0[:, None, :, :] * cell_v_grads_JxW_n0, axis=(0, -1))
            val1_1 = np.sum(tmp1_1[:, None, :, :] * cell_v_grads_JxW_n1, axis=(0, -1))
            val1_2 = np.sum(tmp1_2[:, None, :, :] * cell_v_grads_JxW_n2, axis=(0, -1))
            val1_3 = np.sum(tmp1_3[:, None, :, :] * cell_v_grads_JxW_n3, axis=(0, -1))
            val1_4 = np.sum(tmp1_4[:, None, :, :] * cell_v_grads_JxW_n4, axis=(0, -1))
            val1_5 = np.sum(tmp1_5[:, None, :, :] * cell_v_grads_JxW_n5, axis=(0, -1))
            

            ##################### (num_nodes_p, vec_p)
            tmp2_0 = MnV * vmap_f_local_grad(n0, n_sum) # (num_quads,)
            tmp2_1 = MnV * vmap_f_local_grad(n1, n_sum) # (num_quads,)
            tmp2_2 = MnV * vmap_f_local_grad(n2, n_sum) # (num_quads,)
            tmp2_3 = MnV * vmap_f_local_grad(n3, n_sum) # (num_quads,)
            tmp2_4 = MnV * vmap_f_local_grad(n4, n_sum) # (num_quads,)
            tmp2_5 = MnV * vmap_f_local_grad(n5, n_sum) # (num_quads,)

            
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val2_0 = np.sum(tmp2_0[:, None, None] * self.fe_n0.shape_vals[:, :, None] * cell_JxW_n0[:, None, None], axis=0)
            val2_1 = np.sum(tmp2_1[:, None, None] * self.fe_n1.shape_vals[:, :, None] * cell_JxW_n1[:, None, None], axis=0)
            val2_2 = np.sum(tmp2_2[:, None, None] * self.fe_n2.shape_vals[:, :, None] * cell_JxW_n2[:, None, None], axis=0)
            val2_3 = np.sum(tmp2_3[:, None, None] * self.fe_n3.shape_vals[:, :, None] * cell_JxW_n3[:, None, None], axis=0)
            val2_4 = np.sum(tmp2_4[:, None, None] * self.fe_n4.shape_vals[:, :, None] * cell_JxW_n4[:, None, None], axis=0)
            val2_5 = np.sum(tmp2_5[:, None, None] * self.fe_n5.shape_vals[:, :, None] * cell_JxW_n5[:, None, None], axis=0)

            
            
            dt = self.params['dt']
            tmp3_0 = (n0 - n0_old) / dt # (num_quads,)
            tmp3_1 = (n1 - n1_old) / dt # (num_quads,)
            tmp3_2 = (n2 - n2_old) / dt # (num_quads,)
            tmp3_3 = (n3 - n3_old) / dt # (num_quads,)
            tmp3_4 = (n4 - n4_old) / dt # (num_quads,)
            tmp3_5 = (n5 - n5_old) / dt # (num_quads,)

            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val3_0 = np.sum(tmp3_0[:, None, None] * self.fe_n0.shape_vals[:, :, None] * cell_JxW_n0[:, None, None], axis=0)
            val3_1 = np.sum(tmp3_1[:, None, None] * self.fe_n1.shape_vals[:, :, None] * cell_JxW_n1[:, None, None], axis=0)
            val3_2 = np.sum(tmp3_2[:, None, None] * self.fe_n2.shape_vals[:, :, None] * cell_JxW_n2[:, None, None], axis=0)
            val3_3 = np.sum(tmp3_3[:, None, None] * self.fe_n3.shape_vals[:, :, None] * cell_JxW_n3[:, None, None], axis=0)
            val3_4 = np.sum(tmp3_4[:, None, None] * self.fe_n4.shape_vals[:, :, None] * cell_JxW_n4[:, None, None], axis=0)
            val3_5 = np.sum(tmp3_5[:, None, None] * self.fe_n5.shape_vals[:, :, None] * cell_JxW_n5[:, None, None], axis=0)

            val0 = val1_0 + val2_0 + val3_0 
            val1 = val1_1 + val2_1 + val3_1
            val2 = val1_2 + val2_2 + val3_2 
            val3 = val1_3 + val2_3 + val3_3 
            val4 = val1_4 + val2_4 + val3_4 
            val5 = val1_5 + val2_5 + val3_5 



            weak_form = [val0, val1, val2, val3, val4, val5] 

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel

    def set_params(self, params):
        # Override base class method.
        sol_n0_old,  sol_n1_old, sol_n2_old, sol_n3_old, sol_n4_old, sol_n5_old = params
        
        ## Hu: TODO: These terms are not needed
        sol_old_list = [sol_n0_old[self.fe_n0.cells], sol_n1_old[self.fe_n1.cells], sol_n2_old[self.fe_n2.cells], \
                        sol_n3_old[self.fe_n3.cells], sol_n4_old[self.fe_n4.cells], sol_n5_old[self.fe_n5.cells]]
                        

        n_old_quad_list = [self.fe_n0.convert_from_dof_to_quad(sol_n0_old)[:, :, 0],
                           self.fe_n1.convert_from_dof_to_quad(sol_n1_old)[:, :, 0],
                           self.fe_n2.convert_from_dof_to_quad(sol_n2_old)[:, :, 0],
                           self.fe_n3.convert_from_dof_to_quad(sol_n3_old)[:, :, 0],
                           self.fe_n4.convert_from_dof_to_quad(sol_n4_old)[:, :, 0],
                           self.fe_n5.convert_from_dof_to_quad(sol_n5_old)[:, :, 0]]


        self.internal_vars = [sol_n0_old[self.fe_n0.cells],
                              sol_n1_old[self.fe_n1.cells],
                              sol_n2_old[self.fe_n2.cells],
                              sol_n3_old[self.fe_n3.cells],
                              sol_n4_old[self.fe_n4.cells],
                              sol_n5_old[self.fe_n5.cells], 
                              self.fe_n0.convert_from_dof_to_quad(sol_n0_old)[:, :, 0],
                              self.fe_n1.convert_from_dof_to_quad(sol_n1_old)[:, :, 0],
                              self.fe_n2.convert_from_dof_to_quad(sol_n2_old)[:, :, 0],
                              self.fe_n3.convert_from_dof_to_quad(sol_n3_old)[:, :, 0],
                              self.fe_n4.convert_from_dof_to_quad(sol_n4_old)[:, :, 0],
                              self.fe_n5.convert_from_dof_to_quad(sol_n5_old)[:, :, 0]]


def compute_max_op_from_sol_list_jax(sol_list, threshold=0.01):
    sol_list = [np.squeeze(s) for s in sol_list]

    op_array = np.stack(sol_list, axis=1)

    max_op_idx = np.argmax(op_array, axis=1)
    op_sum = np.sum(op_array, axis=1)
    max_op_idx = np.where(op_sum < threshold, -1, max_op_idx)

    return max_op_idx


def save_sols(problem, sol_list, step):
    vtk_path_p0 = os.path.join(vtk_dir, f'p0_{step:06d}.vtu')

    max_op_idx = compute_max_op_from_sol_list_jax(sol_list, threshold=0.01)

    save_sol(problem.fe_n0, sol_list[0], vtk_path_p0, point_infos=[('n0', sol_list[0]), ('n1', sol_list[1]), ('n2', sol_list[2]),
                                                                   ('n3', sol_list[3]), ('n4', sol_list[4]), ('n5', sol_list[5]),
                                                                   ('obj_id', max_op_idx)])



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
    num_ops = 6

    # ### Hu: [n0, n1, n2, n3, n4, n5]
    problem = AllenCahn([mesh] * num_ops, vec=[1] * num_ops, dim=2, ele_type=[ele_type] * num_ops, \
        additional_info=[params])


    points = problem.fe_n0.points
    
    
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

    # 15 index for definition of 6 order params
    # (num_nucli, )
    index_list  = np.array([0, 1, 2, 3, 4] * 3)


    # 15 centers, normalized coordinates (0~1) -> (15, 2) for 6 order params (0~5)
    # (num_nucli, 2)
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

    
    sol_list = [sol_IC_nucl[0].reshape(problem.fe_n0.num_total_nodes, problem.fe_n0.vec), 
                sol_IC_nucl[1].reshape(problem.fe_n1.num_total_nodes, problem.fe_n1.vec),
                sol_IC_nucl[2].reshape(problem.fe_n2.num_total_nodes, problem.fe_n2.vec),
                sol_IC_nucl[3].reshape(problem.fe_n3.num_total_nodes, problem.fe_n3.vec),
                sol_IC_nucl[4].reshape(problem.fe_n4.num_total_nodes, problem.fe_n4.vec),
                sol_IC_nucl[5].reshape(problem.fe_n5.num_total_nodes, problem.fe_n5.vec)]
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
