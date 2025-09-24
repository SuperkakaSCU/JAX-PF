"""
Implicit finite element solver for coupled Allen Cahn & Cahn Hilliard & Momentum balance
Mg precipitate of single varient considering WBM model
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
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh, box_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem
from jax_fem import logger


## Hu: Explicit solver for [c, mu, n1]
from applications.phaseField.precipitation_WBM.implicit_fem.phase_field import PhaseField
## Hu: Implicit solver for [u]
from applications.phaseField.precipitation_WBM.implicit_fem.u_field import Inclusion

from jax import config

config.update("jax_enable_x64", True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)

def compute_max_op_from_sol_list_jax(sol_list, threshold=0.01):
    sol_list = [np.squeeze(s) for s in sol_list]

    op_array = np.stack(sol_list, axis=1)

    max_op_idx = np.argmax(op_array, axis=1)
    op_sum = np.sum(op_array, axis=1)
    max_op_idx = np.where(op_sum < threshold, -1, max_op_idx)

    return max_op_idx


def save_sols(problem, sol_phaseField_list, sol_u_list, step):
    vtk_path_p = os.path.join(vtk_dir, f'p_{step:06d}.vtu')    
    ## Hu: [c, mu, n1]
    sol_c, sol_mu, sol_n1, sol_n2, sol_n3 = sol_phaseField_list 
    ## Hu: [ux, uy, uz]
    sol_u = sol_u_list[0]

    max_op_idx = compute_max_op_from_sol_list_jax([sol_n1, sol_n2, sol_n3], threshold=0.01)
    save_sol(problem.fe_u, sol_u, vtk_path_p, point_infos=[('c', sol_c), ('n1', sol_n1), ('n2', sol_n2), ('n3', sol_n3), ('obj_id', max_op_idx)])


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params_KKS.json')
    params = json_parse(json_file)


    dt = params['dt']
    t_OFF = params['t_OFF']
    Lx = params['Lx']
    Ly = params['Ly']
    Lz = params['Lz']
    nx = params['nx']
    ny = params['ny']
    nz = params['nz']


    # ele_type = 'HEX8'  # 3D polynomial of the element: 1
    ele_type = 'QUAD4'  # 2D polynomial of the element: 1
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)
    # meshio_mesh = box_mesh(Nx=nx, Ny=ny, Nz=nz, domain_x=Lx, domain_y=Ly, domain_z=Lz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


    ## Hu: Sizes of domain 
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    # Lz = np.max(mesh.points[:, 2])
    print("Lx: {0}, Ly:{1}".format(Lx, Ly)) 

    
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    # def front(point):
    #     return np.isclose(point[2], Lz, atol=1e-5)

    # def back(point):
    #     return np.isclose(point[2], 0., atol=1e-5)


    # location_fns = [left, right, back, front]


    ## Hu: Define Dirichlet B.C.
    def zero_dirichlet_val(point):
        return 0.


    dirichlet_bc_info_u = [[left, right, bottom, top, left, right, bottom, top],
                         [0, 0, 0, 0, 1, 1, 1, 1],
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, \
                          zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]


    num_order = 3

    ### Hu: [c, mu, n1, n2, n3]
    phaseField = PhaseField(mesh=[mesh, mesh, mesh, mesh, mesh], vec=[1, 1, 1, 1, 1], dim=2, ele_type=[ele_type, ele_type, ele_type, ele_type, ele_type],  additional_info=[params])

    ### Hu: [u] - [ux, uy, uz]
    problem_u = Inclusion(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info_u, additional_info=[params])
    
    
    ### Hu: Positions of [c, mu, n1, n2, n3]
    points = phaseField.fe_c.points
    

    ### Hu: set initial conditions for phaseField [c, n1, n2, n3]
    ## Hu: IC for c - ((nx+1)*(nx+1), vec)
    scalar_c_IC = np.ones((phaseField.fes[0].num_total_nodes, phaseField.fes[0].vec))*0.04


    ## Hu: IC for mu - ((nx+1)*(nx+1), vec)
    scalar_mu_IC = np.zeros((phaseField.fes[1].num_total_nodes, phaseField.fes[1].vec))
    

    ## Hu: IC for n1, n2, n3 - ((nx+1)*(nx+1), vec)
    def set_ICs(sol_IC, p, domain_size, index, center, rad):        
        p = p.reshape(nx+1, ny+1, phaseField.fes[1].dim)

        scaled_centers = center * domain_size

        dist = np.linalg.norm(p - scaled_centers, axis=2)  
        
        sol_IC = sol_IC.at[index].set(0.5 * (1.0 - np.tanh((dist - rad) / (Lx/nx))))
       
        return sol_IC


    ## Work on each nucleation
    vmap_set_ICs = jax.jit(jax.vmap(set_ICs, in_axes=(None, None, None, 0, 0, 0)))

    ## Initialization of 3 order params (0~2)
    # (num_order, nx, ny)
    sol_IC = np.array([np.zeros((nx+1, ny+1))] * 3)

    # 4 index for definition of 4 order params
    # (num_nucli, )
    index_list  = np.array([0, 0, 1, 2])

    # 15 centers, normalized coordinates (0~1)
    # (num_nucli, 2)
    center_list = np.array([[1.0 / 3.0, 1.0 / 3.0], 
                            [2.0 / 3.0, 2.0 / 3.0], 
                            [3.0 / 4.0, 1.0 / 4.0], 
                            [1.0 / 4.0, 3.0 / 4.0]])

    # (num_nucli, )
    rad_list = np.array([Lx/16.0]*4)

    domain_size = np.array([Lx, Ly]) 

    # (num_nucli, num_vars, nx+1, nx+1)
    sol_IC_nucl = vmap_set_ICs(sol_IC, points, domain_size, index_list, center_list, rad_list)
    
    # (num_order, nx+1, nx+1)    
    sol_IC_nucl = np.sum(sol_IC_nucl, axis=0)
    sol_IC_nucl = np.minimum(sol_IC_nucl, 0.999)
    sol_IC_nucl = sol_IC_nucl.reshape(num_order, -1)
    

    ## phaseField [c, n1, n2, n3]
    sol_phaseField_list = [scalar_c_IC,
                           scalar_mu_IC,
                           sol_IC_nucl[0].reshape(phaseField.fe_n1.num_total_nodes, phaseField.fe_n1.vec), 
                           sol_IC_nucl[1].reshape(phaseField.fe_n2.num_total_nodes, phaseField.fe_n2.vec),
                           sol_IC_nucl[2].reshape(phaseField.fe_n3.num_total_nodes, phaseField.fe_n3.vec)]


    
    ### Hu: pseudo IC for displacement u - ((nx+1)*(nx+1)*(nx+1), vec)
    sol_u_list = [np.zeros((problem_u.fes[0].num_total_nodes, problem_u.fes[0].vec))]


    ############
    ## Hu: Pass pseudo ICs of u to phase field solver
    ############
    sol_u_old = sol_u_list[0]
    sol_u_for_PF_list = [sol_u_old[problem_u.fe_u.cells], problem_u.fe_u.convert_from_dof_to_quad(sol_u_old)]
        
    sol_list = [sol_phaseField_list, sol_u_for_PF_list]

   
    save_sols(problem_u, sol_phaseField_list, sol_u_list, 0)
    
    nIter = int(t_OFF/dt)
    for i in range(nIter + 1):
        print(f"\nStep {i + 1} in {nIter + 1}, time = {(i + 1)*dt}")

        print("**************")
        print(f"**Update u at Step {i + 1} based on Step {i}'s PF result: [c, mu, n1, n2, n3]**")
        sol_c_old, sol_mu_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list
        
        sol_phaseField_old_list = [sol_c_old[phaseField.fe_c.cells],
                                   sol_n1_old[phaseField.fe_n1.cells], sol_n2_old[phaseField.fe_n2.cells], sol_n3_old[phaseField.fe_n3.cells]]                      
        
        quad_phaseField_old_list = [phaseField.fe_c.convert_from_dof_to_quad(sol_c_old)[:, :, 0],
                                    phaseField.fe_n1.convert_from_dof_to_quad(sol_n1_old)[:, :, 0],
                                    phaseField.fe_n2.convert_from_dof_to_quad(sol_n2_old)[:, :, 0],
                                    phaseField.fe_n3.convert_from_dof_to_quad(sol_n3_old)[:, :, 0],]

        #### TODO: sol_u_list[0] is not useful
        print("Pass [c, mu, n1, n2, n3] to u")
        problem_u.set_params([sol_u_list[0], sol_phaseField_old_list, quad_phaseField_old_list])

        print(f"Start solving u at Step {i + 1} using implicit solver...")
        # sol_u_list = solver(problem_u, solver_options={'jax_solver': {}, 'initial_guess': sol_u_list, 'tol': 1.0e-7}) 
        sol_u_list = solver(problem_u, solver_options={'umfpack_solver': {}, 'initial_guess': sol_u_list, 'tol': 1.0e-7}) 

        print(np.max(sol_u_list[0]), np.min(sol_u_list[0]))
        

        
        print("**************")
        print(f"**Update PF result: [c, mu, n1, n2, n3] at Step {i + 1} based on Step {i + 1}'s u field**")
        print(f"Pass u at Step {i + 1} to [c, mu, n1, n2, n3]")
        sol_u_old = sol_u_list[0]
        sol_u_for_PF_list = [sol_u_old[problem_u.fe_u.cells], problem_u.fe_u.convert_from_dof_to_quad(sol_u_old)]

        ## Hu: Update self.internal variable
        phaseField.set_params([sol_phaseField_list, sol_u_for_PF_list])



        print(f"Solve [c, mu, n1, n2, n3] at Step {i + 1} using implicit solver based on [c, mu, n1, n2, n3] at Step {i} and [u] at Step {i + 1}")
        # sol_phaseField_list = solver(phaseField, solver_options={'jax_solver': {}, 'initial_guess': sol_phaseField_list, 'tol': 1.0e-6})   
        sol_phaseField_list = solver(phaseField, solver_options={'umfpack_solver': {}, 'initial_guess': sol_phaseField_list, 'tol': 1.0e-6})  

        logger.debug(f"max of c dofs = {np.max(sol_phaseField_list[0])}, min of c dofs = {np.min(sol_phaseField_list[0])}")
        logger.debug(f"max of n1 dofs = {np.max(sol_phaseField_list[2])}, min of n1 dofs = {np.min(sol_phaseField_list[2])}")
        logger.debug(f"max of n2 dofs = {np.max(sol_phaseField_list[3])}, min of n2 dofs = {np.min(sol_phaseField_list[3])}")
        logger.debug(f"max of n3 dofs = {np.max(sol_phaseField_list[4])}, min of n3 dofs = {np.min(sol_phaseField_list[4])}")

        if (i + 1) % 100 == 0:
            print(f"**Store the result of [c, mu, n, u] at Step {i + 1}**" ) 
            save_sols(problem_u, sol_phaseField_list, sol_u_list, i + 1)

    sol_c_final, sol_mu_final, sol_n1_final, sol_n2_final, sol_n3_final = sol_phaseField_list
    onp.save(os.path.join(output_dir, "sol_c_final.npy"), sol_c_final)
    onp.save(os.path.join(output_dir, "sol_n1_final.npy"), sol_n1_final)
    onp.save(os.path.join(output_dir, "sol_n2_final.npy"), sol_n2_final)
    onp.save(os.path.join(output_dir, "sol_n3_final.npy"), sol_n3_final)

    # exit()


if __name__ == '__main__':
    import time
    start_time = time.time()
    simulation()
    end_time = time.time()

    print("This is implicit:", end_time - start_time)
