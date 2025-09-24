"""
Implicit finite element solver for coupling Allen Cahn and Cahn Hilliard
Note that we don't have the gradient term for the composition, i.e., gradient c terms
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


 

class CoupledACCH(Problem):
    def custom_init(self, params):
        # Hu: phase field variable of [c, n]
        self.fe_c = self.fes[0]
        self.fe_n = self.fes[1]

        self.params = params


    def get_universal_kernel(self):
        ### Free energy for each phase
        def f_local(c):
            fa = (-1.6704 - 4.776 * c + 5.1622 * c * c - 2.7375 * c * c * c + 1.3687 * c * c * c * c)
            fb = (5.0 * c * c - 5.9746 * c - 1.5924)
            return fa, fb
        vmap_f_local = jax.jit(jax.vmap(f_local))

        # first derivative
        # def f_local_c(c):
        #     fac  = (-4.776 + 10.3244 * c - 8.2125 * c * c + 5.4748 * c * c * c)
        #     fbc  = (10.0 * c - 5.9746)
        #     return fac, fbc

        # vmap_f_local_c = jax.jit(jax.vmap(f_local_c))

        # Second derivative
        # def f_local_cc(c):
        #     facc = (10.3244 - 16.425 * c + 16.4244 * c * c)
        #     fbcc = 10.0
        #     return facc, fbcc

        # vmap_f_local_cc = jax.jit(jax.vmap(f_local_cc))

        def vmap_f_local_c(c):
            grad_fa = jax.jit(jax.vmap(jax.grad(lambda c: f_local(c)[0])))
            grad_fb = jax.jit(jax.vmap(jax.grad(lambda c: f_local(c)[1])))
            fac = grad_fa(c)
            fbc = grad_fb(c)
            return fac, fbc
        
        
        def vmap_f_local_cc(c):
            grad_grad_fa = jax.jit(jax.vmap(jax.grad(jax.grad(lambda c: f_local(c)[0]))))
            grad_grad_fb = jax.jit(jax.vmap(jax.grad(jax.grad(lambda c: f_local(c)[1]))))
            facc = grad_grad_fa(c)
            fbcc = grad_grad_fb(c)
            return facc, fbcc


        ### Interpolation function
        def h_local(n):
            h = (10.0 * n * n * n - 15.0 * n * n * n * n + 6.0 * n * n * n * n * n)
            return h
        vmap_h_local = jax.jit(jax.vmap(h_local))


        # first derivative
        def h_local_n(n):
            h_n = (30.0 * n * n - 60.0 * n * n * n + 30.0 * n * n * n * n)
            return h_n
        vmap_h_local_n = jax.jit(jax.vmap(h_local_n))



        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            """
            Handles the weak form with one cell.
            Assume trial function (c, n), test function (q, w)

            cell_sol_flat: (num_nodes*vec + ...,)
            cell_sol_list: [(num_nodes, vec), ...]
            x: (num_quads, dim)  ->  physical_quad_points
            cell_shape_grads: (num_quads, num_nodes + ..., dim)
            cell_JxW: (num_vars, num_quads)
            cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            """

            # sol_old_list, n_old_quad_list = cell_internal_vars
            cell_sol_c_old,  cell_sol_n_old, c_old, n_old = cell_internal_vars
            

            #### Hu: Unassemble the values to different variables
            ## Hu: [c, n] - [(num_nodes_c, vec_c), (num_nodes_n, vec_n)]
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_c, cell_sol_n = cell_sol_list
            

            ## Hu: cell_shape_grads: (num_quads, num_nodes + ..., dim) 
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_c, cell_shape_grads_n = cell_shape_grads_list


            ## Hu: cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_c, cell_v_grads_JxW_n = cell_v_grads_JxW_list


            ## cell_JxW: (num_vars, num_quads)
            cell_JxW_c = cell_JxW[0]
            cell_JxW_n = cell_JxW[1]


            # (1, num_nodes_p, vec_p, 1) * (num_quads, num_nodes_p, 1, dim) -> (num_quads, num_nodes_p, vec_p, dim)
            c_grads = np.sum(cell_sol_c[None, :, :, None] * cell_shape_grads_c[:, :, None, :], axis=1) # (num_quads, vec_p, dim) -> (4, 1, 2)
            n_grads = np.sum(cell_sol_n[None, :, :, None] * cell_shape_grads_n[:, :, None, :], axis=1) # (num_quads, vec_p, dim) -> (4, 1, 2)

            c = np.sum(cell_sol_c[None, :, :] * self.fe_c.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)
            n = np.sum(cell_sol_n[None, :, :] * self.fe_n.shape_vals[:, :, None], axis=1)[:, 0] # (num_quads,)

            c_grads_old = np.sum(cell_sol_c_old[None, :, :, None] * cell_shape_grads_c[:, :, None, :], axis=1) # (num_quads, vec_p, dim) -> (4, 1, 2)
            n_grads_old = np.sum(cell_sol_n_old[None, :, :, None] * cell_shape_grads_n[:, :, None, :], axis=1) # (num_quads, vec_p, dim) -> (4, 1, 2)

            # (num_quads)
            fa, fb = vmap_f_local(c)
            fac, fbc = vmap_f_local_c(c)
            facc, fbcc = vmap_f_local_cc(c)

            # (num_quads)
            h = vmap_h_local(n)
            h_n = vmap_h_local_n(n)


            ######################
            ## This is phase field variable weak form
            ######################
            dt = self.params['dt']

            ## Hu: p & p_old: (num_quads,)
            tmp3_c = (c - c_old) / dt # (num_quads,)
            tmp3_n = (n - n_old) / dt # (num_quads,)
            
            # (num_quads, 1, vec_p) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val3_c = np.sum(tmp3_c[:, None, None] * self.fe_c.shape_vals[:, :, None] * cell_JxW_c[:, None, None], axis=0)
            val3_n = np.sum(tmp3_n[:, None, None] * self.fe_n.shape_vals[:, :, None] * cell_JxW_n[:, None, None], axis=0)


            ######################
            ## This is phase field variable weak form of Cahn-Hilliard 
            ######################
            ##################### (num_nodes_p, vec_p)
            McV = self.params['McV'] # Mc in Eqs - The CH mobility
            # (num_quads)
            tmp1_c_term = McV * ((1.0 - h)*facc + h*fbcc)
            ## Hu: (num_quads, 1, 1) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            tmp1_c = tmp1_c_term[:, None, None] * c_grads
            
            ## Hu: (num_quads, 1, vec_p, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes_p, vec_p)
            val1_c = np.sum(tmp1_c[:, None, :, :] * cell_v_grads_JxW_c, axis=(0, -1))


            # (num_quads)
            tmp2_c_term = McV * ((fbc - fac) * h_n)
            ## Hu: (num_quads, 1, 1) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            tmp2_c = tmp2_c_term[:, None, None] * n_grads
            
            ## Hu: (num_quads, 1, vec_p, dim) * (num_quads, num_nodes, 1, dim) -> (num_nodes_p, vec_p)
            val2_c = np.sum(tmp2_c[:, None, :, :] * cell_v_grads_JxW_c, axis=(0, -1))

            val_c = val1_c + val2_c + val3_c



            ######################
            ## This is phase field variable weak form of Allen-Cahn
            ######################
            ## Hu: (1,) * (num_quads, vec_p, dim) -> (num_quads, vec_p, dim)
            MnV = self.params['MnV'] # The AC mobility, MnV
            KnV = self.params['KnV'] # Gradient energy coefficient
            # (num_quads)
            # tmp1_n = MnV * ((fb - fa)*h_n)
            tmp1_n = MnV * ((fbc - fac)*h_n)
            # (num_quads, 1, 1) * (num_quads, num_nodes_p, 1) * (num_quads, 1, 1) -> (num_nodes_p, vec_p)
            val1_n = np.sum(tmp1_n[:, None, None] * self.fe_n.shape_vals[:, :, None] * cell_JxW_n[:, None, None], axis=0)

            # (num_quads, vec_p, dim)
            tmp2_n = MnV * KnV * n_grads # (num_quads, vec_p, dim)
            val2_n = np.sum(tmp2_n[:, None, :, :] * cell_v_grads_JxW_n, axis=(0, -1))
            
            val_n = val1_n + val2_n + val3_n
            

            weak_form = [val_c, val_n] 

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        # return jax.jit(universal_kernel)
        return universal_kernel

    def set_params(self, params):
        # Override base class method.
        sol_c_old,  sol_n_old = params
       
        self.internal_vars = [sol_c_old[self.fe_c.cells],
                              sol_n_old[self.fe_n.cells],
                              self.fe_c.convert_from_dof_to_quad(sol_c_old)[:, :, 0],
                              self.fe_n.convert_from_dof_to_quad(sol_n_old)[:, :, 0]]


def compute_max_op_from_sol_list_jax(sol_list, threshold=0.01):
    sol_list = [np.squeeze(s) for s in sol_list]

    op_array = np.stack(sol_list, axis=1)

    max_op_idx = np.argmax(op_array, axis=1)
    op_sum = np.sum(op_array, axis=1)
    max_op_idx = np.where(op_sum < threshold, -1, max_op_idx)

    return max_op_idx


def save_sols(problem, sol_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:06d}.vtu')

    save_sol(problem.fe_n, sol_list[0], vtk_path_p, point_infos=[('c', sol_list[0]), ('n', sol_list[1])])



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
    print("Lx:{0}, Ly:{1}".format(Lx, Ly))

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


    location_fns = [left, right, top, bottom]


    ### Hu: [c, n]
    problem = CoupledACCH(mesh=[mesh, mesh], vec=[1, 1], dim=2, ele_type=[ele_type, ele_type],  additional_info=[params])


    
    points = problem.fe_n.points
    
    

    def set_ICs(sol_IC, p, domain_size, index, center, rad):
        
        p = p.reshape(nx+1, ny+1, problem.fes[0].dim)

        scaled_centers = center * domain_size

        dist = np.linalg.norm(p - scaled_centers, axis=2)
        

        rad = rad*domain_size[0]

        phi = 0.5 * (1.0 - np.tanh((dist - rad) / 1.0))

        # phase field c
        def case0(_): return sol_IC.at[index].set(0.125 * phi)
        # phase field n
        def case1(_): return sol_IC.at[index].set(phi)

        def default(_): return sol_IC.at[index].set(phi)

        sol_IC = jax.lax.cond(index == 0, case0,
                          lambda _: jax.lax.cond(index == 1, case1, default, None),
                          operand=None)
        return sol_IC


    vmap_set_ICs = jax.jit(jax.vmap(set_ICs, in_axes=(None, None, None, 0, 0, 0)))
    

    ## Initialization of PF variables [c, n]
    matrix_concentration = params['matrix_concentration']
    sol_IC = np.array([np.zeros((nx+1, ny+1)), np.zeros((nx+1, ny+1))])
    
    ### [c, c, n, n]
    index_list  = np.array([0, 0, 1, 1])
    

    ## Centers for points 1 and 2 for both c and n, normalized coordinates (0~1)
    # (num_nucli, 2)
    center_list = np.array([[0.333333, 0.333333], [0.75, 0.75], [0.333333, 0.333333], [0.75, 0.75]])

    ## Radii for points 1 and 2 for both c and n, normalized coordinates (0~1)
    # (num_nucli, )
    rad_list = np.array([20.0/100.0, 8.333333/100.0, 20.0/100.0, 8.333333/100.0])

    domain_size = np.array([Lx, Ly])  # shape: (3,)

    
    # (num_nucli, num_vars, nx+1, nx+1)
    sol_IC_nucl = vmap_set_ICs(sol_IC, points, domain_size, index_list, center_list, rad_list)  
    sol_IC_nucl = np.sum(sol_IC_nucl, axis=0)
    sol_IC_nucl = sol_IC_nucl.at[0].set(sol_IC_nucl[0] + matrix_concentration)


    sol_IC_nucl = sol_IC_nucl.reshape(problem.num_vars, -1)

    
    sol_list = [sol_IC_nucl[0].reshape(problem.fe_n.num_total_nodes, problem.fe_n.vec), 
                sol_IC_nucl[1].reshape(problem.fe_c.num_total_nodes, problem.fe_c.vec)]
    save_sols(problem, sol_list, 0)


    nIter = int(t_OFF/dt)
    print("nIter", nIter)

    for i in range(nIter + 1):
    # for i in range(2):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")
        
        problem.set_params(sol_list)

        sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list, 'tol': 1e-7})   
        # sol_list = solver(problem, solver_options={'petsc_solver': {}, 'initial_guess': sol_list})   
        # sol_list = solver(problem, solver_options={'umfpack': {}, 'initial_guess': sol_list})   

        
        if (i + 1) % 100 == 0:
            save_sols(problem, sol_list, i + 1)



if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is implicit solver for coupled Allen-Cahn and Cahn-Hilliard:", end_time - start_time)
