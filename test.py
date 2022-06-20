import sympy
import numpy as np
from sympy.matrices import Matrix, zeros
from sympy import Rational, symbols, integrate, nsimplify, diff, lambdify
from functools import lru_cache, partial
import os.path
import os
import jax.numpy as jnp
import jax
from jax.experimental import sparse as jsparse
from jax.experimental.sparse import sparsify
from scipy import sparse
from scipy.sparse import dok_matrix
from jax.scipy.linalg import lu_factor
import timeit
import time
from jax import config, jit
from jax.scipy.sparse.linalg import *

config.update("jax_enable_x64", True)

from basisfunctions import (
	legendre_poly,
	FE_poly,
	node_locations,
	num_elements,
	num_elements_FE,
)
import sparsesolve

def get_bottom_indices(order):
	if order == 1 or order == 0:
		return jnp.asarray([0], dtype=int)
	if order == 2:
		return jnp.asarray([0, 1, 7], dtype=int)
	if order == 3:
		return jnp.asarray([0, 1, 2, 10, 11], dtype=int)
	if order == 4:
		return jnp.asarray([0, 1, 2, 3, 9, 14, 15, 16], dtype=int)
	raise Exception


def is_bottom_element(order, k):
	arr = get_bottom_indices(order)
	if order == 1 or order == 0:
		if k in arr:
			return True
	elif order == 2:
		if k in arr:
			return True
	elif order == 3:
		if k in arr:
			return True
	elif order == 4:
		if k in arr:
			return True
	else:
		raise Exception
	return False


def convert_to_bottom_indices(T, order):
	def convert_to_bottom_index(index):
		if order == 1 or order == 0:
			if index == 0:
				return 0
			else:
				raise Exception
		if order == 2:
			if index == 0 or index == 1:
				return index
			if index == 7:
				return 2
			else:
				raise Exception
		if order == 3:
			if index == 0 or index == 1 or index == 2:
				return index
			if index == 10 or index == 11:
				return index - 7
			else:
				raise Exception
		if order == 4:
			if index == 0 or index == 1 or index == 2 or index == 3:
				return index
			if index == 9:
				return 4
			if index == 14 or index == 15 or index == 16:
				return index - 9
			else:
				raise Exception

	T = np.asarray(T, dtype=int)
	T_new = np.zeros(T.shape)
	T_new[:, 0] = T[:, 0]
	T_new[:, 1] = T[:, 1]
	for i in range(T.shape[0]):
		T_new[i, 2] = convert_to_bottom_index(T[i, 2])
	return jnp.asarray(T_new, dtype=int)


def load_assembly_matrix(basedir, nx, ny, order):
	def create_assembly_matrix(nx, ny, order):
		"""
		Generates an assembly matrix which converts the
		local/element matrices to the global matrices
		"""
		table = {}
		nodes = node_locations(order)
		num_elem = nodes.shape[0]

		def lookup_table(ijk):
			i, j, k = ijk
			x, y = nodes[k, :]
			i_l = (i - 1) % nx
			i_r = (i + 1) % nx
			j_b = (j - 1) % ny
			j_t = (j + 1) % ny
			if (i, j, x, y) in table:
				return table[(i, j, x, y)]
			elif (i_l, j, x + 2, y) in table:
				return table[(i_l, j, x + 2, y)]
			elif (i_r, j, x - 2, y) in table:
				return table[(i_r, j, x - 2, y)]
			elif (i, j_t, x, y - 2) in table:
				return table[(i, j_t, x, y - 2)]
			elif (i, j_b, x, y + 2) in table:
				return table[(i, j_b, x, y + 2)]
			elif (i_l, j_t, x + 2, y - 2) in table:
				return table[(i_l, j_t, x + 2, y - 2)]
			elif (i_r, j_t, x - 2, y - 2) in table:
				return table[(i_r, j_t, x - 2, y - 2)]
			elif (i_l, j_b, x + 2, y + 2) in table:
				return table[(i_l, j_b, x + 2, y + 2)]
			elif (i_r, j_b, x - 2, y + 2) in table:
				return table[(i_r, j_b, x - 2, y + 2)]
			else:
				return None

		def assign_table(ijk, node_val):
			i, j, k = ijk
			x, y = nodes[k, :]
			table[(i, j, x, y)] = node_val
			return

		node_index = 0
		for j in range(ny):
			for i in range(nx):
				for k in range(num_elem):
					ijk = (i, j, k)
					node_val = lookup_table(ijk)
					if node_val is None:
						node_val = node_index
						node_index += 1
					assign_table(ijk, node_val)

		num_global_elements = max(table.values()) + 1
		M = np.zeros((nx, ny, num_elem), dtype=int)
		T = -np.ones((num_global_elements, 3), dtype=int)

		for i in range(nx):
			for j in range(ny):
				for k in range(num_elem):
					x, y = nodes[k, :]
					gamma = table[(i, j, x, y)]
					M[i, j, k] = gamma
					if T[gamma, 0] == -1 and is_bottom_element(order, k):
						T[gamma, 0] = i
						T[gamma, 1] = j
						T[gamma, 2] = k

		return num_global_elements, M, T

	if os.path.exists(
		"{}/data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
			basedir, nx, ny, order
		)
	):
		num_global_elements = np.load(
			"{}/data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			)
		)
		M = np.load(
			"{}/data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			)
		)
		T = np.load(
			"{}/data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			)
		)
	else:
		num_global_elements, M, T = create_assembly_matrix(nx, ny, order)
		np.save(
			"{}/data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			),
			num_global_elements,
		)
		np.save(
			"{}/data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			),
			M,
		)
		np.save(
			"{}/data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
				basedir, nx, ny, order
			),
			T,
		)
	return num_global_elements, M, T


def load_elementwise_volume(basedir, nx, ny, Lx, Ly, order):
	"""
	Returns the (num_elements x num_elements) matrix
	where the (i,j) component is the elementwise integral
	V_{ij} = int_Omega nabla psi_i nabla psi_j dx dy
	in "local" coordinates.

	Later we will map this matrix to "global" coordinates.
	"""

	def create_elementwise_volume(order):
		basis = FE_poly(order)
		num_elem = basis.shape[0]
		res1 = np.zeros((num_elem, num_elem))
		res2 = np.zeros((num_elem, num_elem))
		for i in range(num_elem):
			for j in range(num_elem):
				expr1 = diff(basis[i], "x") * diff(basis[j], "x")
				res1[i, j] = integrate(expr1, ("x", -1, 1), ("y", -1, 1))
				expr2 = diff(basis[i], "y") * diff(basis[j], "y")
				res2[i, j] = integrate(expr2, ("x", -1, 1), ("y", -1, 1))
		return res1, res2

	dx = Lx / nx
	dy = Ly / ny
	if os.path.exists(
		"{}/data/poissonmatrices/elementwise_volume_{}_1.npy".format(basedir, order)
	):
		res1 = np.load(
			"{}/data/poissonmatrices/elementwise_volume_{}_1.npy".format(basedir, order)
		)
		res2 = np.load(
			"{}/data/poissonmatrices/elementwise_volume_{}_2.npy".format(basedir, order)
		)
	else:
		res1, res2 = create_elementwise_volume(order)
		np.save(
			"{}/data/poissonmatrices/elementwise_volume_{}_1".format(basedir, order),
			res1,
		)
		np.save(
			"{}/data/poissonmatrices/elementwise_volume_{}_2".format(basedir, order),
			res2,
		)
	V = res1 * (dy / dx) + res2 * (dx / dy)
	return V


def load_elementwise_source(basedir, nx, ny, Lx, Ly, order):
	def write_elementwise_source(order):
		FE_basis = FE_poly(order)
		legendre_basis = legendre_poly(order)
		res = np.zeros((FE_basis.shape[0], legendre_basis.shape[0]))
		for i in range(FE_basis.shape[0]):
			for j in range(legendre_basis.shape[0]):
				expr = FE_basis[i] * legendre_basis[j]
				res[i, j] = integrate(expr, ("x", -1, 1), ("y", -1, 1))
		return res

	dx = Lx / nx
	dy = Ly / ny
	if os.path.exists(
		"{}/data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order)
	):
		res = np.load(
			"{}/data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order)
		)
	else:
		res = write_elementwise_source(order)
		np.save(
			"{}/data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order),
			res,
		)
	return res * dx * dy / 4


def load_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
	if os.path.exists(
		"{}/data/poissonmatrices/volume_{}_{}_{}.npz".format(basedir, nx, ny, order)
	):
		sV = sparse.load_npz(
			"{}/data/poissonmatrices/volume_{}_{}_{}.npz".format(basedir, nx, ny, order)
		)
	else:
		V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements)
		sV = sparse.csr_matrix(V)
		sparse.save_npz(
			"{}/data/poissonmatrices/volume_{}_{}_{}.npz".format(
				basedir, nx, ny, order
			),
			sV,
		)
	return sV


def create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
	num_elem = num_elements(order)
	K_elementwise = load_elementwise_volume(basedir, nx, ny, Lx, Ly, order)

	sK = dok_matrix((num_global_elements, num_global_elements))

	for j in range(ny):
		for i in range(nx):
			sK[M[i, j, :][:, None], M[i, j, :][None, :]] += K_elementwise[:, :]
	return sK

"""
def create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
	num_elem = num_elements(order)
	K_elementwise = load_elementwise_volume(basedir, nx, ny, Lx, Ly, order)

	K = np.zeros((num_global_elements, num_global_elements))

	for j in range(ny):
		for i in range(nx):
			K[M[i, j, :][:, None], M[i, j, :][None, :]] += K_elementwise[:, :]
	return K
"""

def get_kernel(order):
	bottom_indices = get_bottom_indices(order)
	K = np.zeros((2, 2, num_elements_FE(order), num_elements_FE(order)))
	if order == 1 or order == 0:
		K[0, 0, 0, 2] = 1.0
		K[1, 0, 0, 3] = 1.0
		K[0, 1, 0, 1] = 1.0
		K[1, 1, 0, 0] = 1.0
	elif order == 2:
		K[0, 0, 0, 4] = 1.0
		K[1, 0, 0, 6] = 1.0
		K[1, 0, 1, 5] = 1.0
		K[0, 1, 0, 2] = 1.0
		K[0, 1, 7, 3] = 1.0
		K[1, 1, 0, 0] = 1.0
		K[1, 1, 1, 1] = 1.0
		K[1, 1, 7, 7] = 1.0
	elif order == 3:
		K[0, 0, 0, 6] = 1.0
		K[1, 0, 0, 9] = 1.0
		K[1, 0, 1, 8] = 1.0
		K[1, 0, 2, 7] = 1.0
		K[0, 1, 0, 3] = 1.0
		K[0, 1, 11, 4] = 1.0
		K[0, 1, 10, 5] = 1.0
		K[1, 1, 0, 0] = 1.0
		K[1, 1, 1, 1] = 1.0
		K[1, 1, 2, 2] = 1.0
		K[1, 1, 10, 10] = 1.0
		K[1, 1, 11, 11] = 1.0
	elif order == 4:
		K[1, 1, 0, 0] = 1.0
		K[1, 1, 1, 1] = 1.0
		K[1, 1, 2, 2] = 1.0
		K[1, 1, 3, 3] = 1.0
		K[1, 1, 9, 9] = 1.0
		K[1, 1, 14, 14] = 1.0
		K[1, 1, 15, 15] = 1.0
		K[1, 1, 16, 16] = 1.0
		K[0, 0, 0, 8] = 1.0
		K[1, 0, 0, 13] = 1.0
		K[1, 0, 1, 12] = 1.0
		K[1, 0, 2, 11] = 1.0
		K[1, 0, 3, 10] = 1.0
		K[0, 1, 0, 4] = 1.0
		K[0, 1, 16, 5] = 1.0
		K[0, 1, 15, 6] = 1.0
		K[0, 1, 14, 7] = 1.0
	else:
		raise Exception
	return jnp.asarray(K)[:, :, bottom_indices, :]

def get_V_sp_K_T_M(nx, ny, order):

	basedir='/home/mcgreivy/sparse_solve_tests'
	Lx = Ly = 1.0

	N_global_elements, M, T = load_assembly_matrix(basedir, nx, ny, order)
	T = convert_to_bottom_indices(T, order)
	S_elem = load_elementwise_source(basedir, nx, ny, Lx, Ly, order)

	K = get_kernel(order) @ S_elem


	sV = load_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
	V_sp = jsparse.BCOO.from_scipy_sparse(sV)

	return V_sp, K, T, M







def get_custom_poisson_solver(V_sp):

    args = V_sp.data, V_sp.indices, V_sp.shape[0]
    kwargs = {"forward": True}
    custom_lu_solve = lambda b: sparsesolve.sparse_solve_prim(b, *args, **kwargs)

    return custom_lu_solve


@sparsify
def f(M, v):
	return M @ v



def get_gmres_solver(V_sp):
	V_func = lambda b: f(V_sp, b)
	def solve(b):
		res, _ = gmres(V_func, b)
		return res
	return solve


def get_cg_solver(V_sp):
	V_func = lambda b: f(V_sp, b)
	def solve(b):
		res, _ = cg(V_func, b)
		return res
	return solve


def get_bicgstab_solver(V_sp):
	V_func = lambda b: f(V_sp, b)
	def solve(b):
		res, _ = bicgstab(V_func, b)
		return res
	return solve



def main():


	nx = ny = 32
	order = 1
	N_test = 5
	key = jax.random.PRNGKey(0)


	V_sp, K, T, M = get_V_sp_K_T_M(nx, ny, order)


	@partial(jax.jit, static_argnums=(1,))
	def solve(xi, lu_solve):
	    xi = xi.at[:, :, 0].add(-jnp.mean(xi[:, :, 0]))
	    xi = jnp.pad(xi, ((1, 0), (1, 0), (0, 0)), mode="wrap")
	    F_ijb = jax.lax.conv_general_dilated(
	        xi[None, ...],
	        K,
	        (1, 1),
	        padding="VALID",
	        dimension_numbers=("NHWC", "HWOI", "NHWC"),
	    )[0]
	    b = -F_ijb[T[:, 0], T[:, 1], T[:, 2]]

	    res = lu_solve(b)
	    res = res - jnp.mean(res)
	    output = res.at[M].get()
	    return output


	def print_output(key, test_solve):
		for i in range(N_test):
			key, _ = jax.random.split(key, 2)
			xi = jax.random.normal(key, shape=(nx, ny, num_elements(order)))
			t1 = time.time()
			output = solve(xi, test_solve).block_until_ready()
			t2 = time.time()
			print("{} milliseconds".format((t2 - t1) * 10e3))


	from jax.lib import xla_bridge
	print(xla_bridge.get_backend().platform)

	#### Test custom LU solve
	#custom_solve = get_custom_poisson_solver(V_sp)
	gmres_solve = get_gmres_solver(V_sp)
	cg_solve = get_cg_solver(V_sp)
	bicgstab_solve = get_bicgstab_solver(V_sp)

	#print("Custom solve")
	#print_output(key, custom_solve)

	print("gmres")
	print_output(key, gmres_solve)

	print("cg")
	print_output(key, cg_solve)

	print("bicgstab")
	print_output(key, bicgstab_solve)

	X = jnp.ones((1000,1000))
	@jax.jit
	def f(x):
		return x @ x
	Y = f(X).block_until_ready()
	t1 = time.time()
	Y = f(X).block_until_ready()
	t2 = time.time()
	print("{} milliseconds for matmul".format((t2 - t1)*1000))

if __name__ == '__main__':
	main()


