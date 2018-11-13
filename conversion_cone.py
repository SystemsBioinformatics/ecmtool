import numpy as np
from helpers import *
from sympy import Matrix
from time import time


def normalize_rows(M):
    row_max = M.max(axis=1)
    return M / np.transpose(np.asarray(np.asmatrix(row_max, dtype='object'), dtype='object'))

def get_rownames(A):
    rownames = []
    for row_index in range(A.shape[0]):
        rownames.append([index for index, value in enumerate(A[row_index, :]) if value != 0])
    return rownames


def deflate_matrix(A, columns_to_keep):
    B = np.ndarray(shape=(0, len(columns_to_keep)), dtype=A.dtype)

    # Return rows that are nonzero after removing unwanted columns
    for row_index in range(A.shape[0]):
        row = A[row_index, columns_to_keep]
        if np.sum(row) != 0:
            B = np.append(B, [row], axis=0)

    return B


def inflate_matrix(A, kept_columns, original_width):
    B = np.zeros(shape=(A.shape[0], original_width), dtype=A.dtype)

    for index, col in enumerate(kept_columns):
        B[:, col] = A[:, index]

    return B


def get_conversion_cone(N, tagged_rows=[], reversible_columns=[], input_metabolites=[], output_metabolites=[],
                        symbolic=True, verbose=False):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).
    :param N: stoichiometry matrix
    :param tagged_rows: list of row numbers (0-based) of metabolites that are tagged as in/outputs ("conversions")
    :param reversible_columns: list of booleans stating whether the reaction at this column is reversible
    :return: matrix with conversion cone "c" as row vectors
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]

    # Compose G of the columns of N
    G = np.transpose(N)

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for metabolite_index in range(G.shape[0]):
        if metabolite_index in reversible_columns:
            G = np.append(G, [-G[metabolite_index, :]], axis=0)

    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
        print('Calculating extreme rays H of inequalities system G')

    # Calculate generating set of the dual of our initial conversion cone C0, C0*
    rays_full = get_extreme_rays_cdd(G)

    # Remove internal metabolites from the rays, since they are all equal to 0 in the conversion cone
    rays_deflated = deflate_matrix(rays_full, tagged_rows)

    H_ineq = np.ndarray(shape=(0, rays_deflated.shape[1]))
    linearities = np.ndarray(shape=(0, rays_deflated.shape[1]))

    # Fill H_ineq with all generating rays that are not part of the nullspace (aka lineality space) of the dual cone C0*.
    # These represent the system of inequalities of our initial conversion cone C0.
    for row in range(rays_deflated.shape[0]):
        if np.all(np.dot(G, rays_full[row, :]) == 0):
            # This is a linearity
            linearities = np.append(linearities, [rays_deflated[row, :]], axis=0)
        else:
            # This is a normal extreme ray
            H_ineq = np.append(H_ineq, [rays_deflated[row, :]], axis=0)


    H_eq = linearities

    # If there are inequalities, apply trick A3 from (Urbanczik, 2005, appendix)
    make_homogeneous = H_ineq.shape[0] > 0

    if verbose and make_homogeneous:
        print('Calculating nullspace A of H_eq')

    A = nullspace(H_eq) if make_homogeneous else None
    H_total = np.dot(H_ineq, A) if make_homogeneous else np.append(H_eq, -H_eq, axis=0)

    # Calculate the extreme rays of the cone C represented by inequalities H_total, resulting in
    # the elementary conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H_total')

    if not H_ineq.shape[0]:
        H_ineq = np.zeros(shape=(1, H_ineq.shape[1]))

    rays = np.asarray(list(get_extreme_rays_efmtool(H_total)))
    rays_deflated = np.transpose(np.dot(A, np.transpose(rays))) if make_homogeneous else rays

    if rays_deflated.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays_deflated, H_ineq

    rays_inflated = inflate_matrix(rays_deflated, tagged_rows, amount_metabolites)

    return rays_inflated, H_ineq


if __name__ == '__main__':
    start = time()

    S = np.asarray([
        [-1, 0, 0],
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, -1],
        [0, 0, 1]])
    c = get_conversion_cone(S, [0, 1, 4], verbose=True)
    print(c)

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
