from time import time

from .helpers import *
from .network import Network, Reaction, Metabolite
from .nullspace import iterative_nullspace
from .custom_redund import drop_redundant_rays, remove_cycles_redund
from .intersect_directly_mpi import independent_rows
from ecmtool import mpi_wrapper


def normalize_rows(M):
    row_max = abs(M.max(axis=1))
    return M / np.transpose(np.asarray(np.asmatrix(row_max, dtype='object'), dtype='object'))


def get_rownames(A):
    rownames = []
    for row_index in range(A.shape[0]):
        rownames.append([index for index, value in enumerate(A[row_index, :]) if value != 0])
    return rownames


def deflate_matrix(A, columns_to_keep):
    reduced_A = A[:, columns_to_keep]

    # Return rows that are nonzero after removing unwanted columns
    B = reduced_A[np.where(np.count_nonzero(reduced_A, axis=1) > 0)[0], :]

    return B


def inflate_matrix(A, kept_columns, original_width):
    B = np.zeros(shape=(A.shape[0], original_width), dtype=A.dtype)

    for index, col in enumerate(kept_columns):
        B[:, col] = A[:, index]

    return B


def get_zero_set(G, equalities):
    dot_products = np.transpose(np.asarray([np.dot(G, equality) for equality in equalities]))
    zero_set = [set([index for index, value in enumerate(row) if value == 0]) for row in dot_products]
    return zero_set


def drop_nonextreme(G, zero_set, verbose=False):
    keep_rows = []

    for row in range(G.shape[0]):
        remove = False
        for other_row in range(G.shape[0]):
            # The current row (reaction) has a zero set that is
            # a proper subset of another row's. This means the other
            # row represents a more extreme vector in the same plane.
            if zero_set[row] < zero_set[other_row]:
                remove = True
                break

        if not remove:
            keep_rows.append(row)


    if verbose:
        print('Removing %d nonextreme rays' % (G.shape[0] - len(keep_rows)))

    return G[keep_rows, :]


def split_columns_semipositively(matrix, columns):
    orig_column_count = matrix.shape[1]
    matrix = split_columns(matrix, columns)
    semipositive_columns = columns + [orig_column_count + index for index in range(len(columns))]

    for row in range(matrix.shape[0]):
        for column in semipositive_columns:
            matrix[row, column] = max(matrix[row, column], 0)

    return matrix


def split_columns(matrix, columns):
    matrix = np.append(matrix, -matrix[:, columns], axis=1)
    return matrix

def calculate_linearities(N, reversible_reactions, external_metabolites, input_metabolites, output_metabolites, verbose=False):
    """
    :param N: stoichiometry matrix
    :param reversible_reactions: list of booleans stating whether the reaction at this column is reversible
    :param external_metabolites: list of row numbers (0-based) of metabolites that are tagged as in/outputs
    :param input_metabolites: list of row numbers (0-based) of metabolites that are tagged as inputs
    :param output_metabolites: list of row numbers (0-based) of metabolites that are tagged as outputs
    :param verbose: print status messages during enumeration
    :return:
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]

    # External metabolites that have no direction specified
    in_out_metabolites = np.setdiff1d(external_metabolites, np.append(input_metabolites, output_metabolites, axis=0))
    added_virtual_metabolites = np.asarray(np.add(range(len(in_out_metabolites)), amount_metabolites), dtype='int')
    extended_external_metabolites = np.append(external_metabolites, added_virtual_metabolites, axis=0)
    in_out_indices = [external_metabolites.index(index) for index in in_out_metabolites]

    if len(external_metabolites) == 0:
        return to_fractions(np.ndarray(shape=(0, N.shape[0])))

    # Compose G of the columns of N
    G = np.transpose(N)

    G_exp = G[:, :]
    G_rev = np.ndarray(shape=(0, G.shape[1]), dtype='object')
    G_irrev = np.ndarray(shape=(0, G.shape[1]), dtype='object')

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_reactions:
            G_exp = np.append(G_exp, [-G[reaction_index, :]], axis=0)
            G_rev = np.append(G_rev, [-G[reaction_index, :]], axis=0)
        else:
            G_irrev = np.append(G_irrev, [G[reaction_index, :]], axis=0)

    if verbose:
        print('Calculating null space of inequalities system G')
    linearities = np.transpose(iterative_nullspace(G, verbose=verbose))

    if linearities.shape[0] == 0:
        linearities = np.ndarray(shape=(0, G.shape[1]))

    linearities_deflated = deflate_matrix(linearities, external_metabolites)

    return linearities, linearities_deflated, G_rev, G_irrev, amount_metabolites, extended_external_metabolites, in_out_indices


def calc_C0_dual_extreme_rays(linearities, G_rev, G_irrev, polco, processes, jvm_mem, path2mplrs, verbose=False):
    """
    :param linearities:
    :param G_rev:
    :param G_irrev:
    :param polco: set to True to make computation with polco instead of mplrs
    :param processes: integer value giving the number of processes
    :param jvm_mem: tuple of integer giving the minimum and maximum number of java VM memory in GB
    :param path2mplrs: absolute path to mplrs binary
    :param verbose:
    :return:
    """
    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
        print('Calculating extreme rays H of inequalities system G')

    # Calculate generating set of the dual of our initial conversion cone C0, C0*
    rays = get_extreme_rays(np.append(linearities, G_rev, axis=0), G_irrev, verbose=verbose, polco=polco,
                            processes=processes, jvm_mem=jvm_mem, path2mplrs=path2mplrs)
    return rays


def calc_H(rays, linearities_deflated, external_metabolites, input_metabolites,
           output_metabolites, in_out_indices, redund_after_polco=True, only_rays=False, verbose=False):
    """
    :param rays:
    :param linearities_deflated:
    :param external_metabolites:
    :param input_metabolites:
    :param output_metabolites:
    :param in_out_indices:
    :param redund_after_polco:
    :param only_rays:
    :param verbose:
    :return:
    """
    if verbose:
        print('Deflating H')
    if rays.shape[0] == 0:
        mp_print('Warning: first polco-application did not give any rays. Check if this is expected behaviour.')
        rays_deflated = rays
    else:
        rays_deflated = deflate_matrix(rays, external_metabolites)

    if verbose:
        print('Expanding H with metabolite direction constraints')
    # Add bidirectional (in- and output) metabolites in reverse direction
    if rays_deflated.shape[0] == 0:
        rays_split = rays_deflated
    else:
        rays_split = split_columns(rays_deflated, in_out_indices) if not only_rays else rays_deflated
    linearities_split = split_columns(linearities_deflated, in_out_indices) if not only_rays else linearities_deflated

    H_ineq = rays_split
    H_eq = linearities_split

    # Add input/output constraints to H_ineq
    if not H_ineq.shape[0]:
        H_ineq = np.zeros(shape=(1, H_ineq.shape[1]))

    identity = to_fractions(np.identity(H_ineq.shape[1]))

    # Bidirectional (in- and output) metabolites.
    # When enumerating only extreme rays, no splitting is done, and
    # thus no other dimensions need to have directionality specified.
    if not only_rays:
        for list_index, inout_metabolite_index in enumerate(in_out_indices):
            index = inout_metabolite_index
            H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

            index = len(external_metabolites) + list_index
            H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

    # Inputs
    for input_metabolite in input_metabolites:
        index = external_metabolites.index(input_metabolite)
        H_ineq = np.append(H_ineq, [-identity[index, :]], axis=0)

    # Outputs
    for output_metabolite in output_metabolites:
        index = external_metabolites.index(output_metabolite)
        H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

    if verbose:
        print('Reducing rows in H by removing redundant rows')

    # Use redundancy-removal to make H_ineq and H_eq smaller
    print("Size of H_ineq before redund:", H_ineq.shape[0], H_ineq.shape[1])
    print("Size of H_eq before redund:", H_eq.shape[0], H_eq.shape[1])
    count_before_ineq = len(H_ineq)
    count_before_eq = len(H_eq)

    if verbose:
        mp_print('Detecting linearities in H_ineq.')
    H_ineq_transpose, cycle_rays = remove_cycles_redund(np.transpose(H_ineq))
    H_ineq = np.transpose(H_ineq_transpose)

    H_eq = np.concatenate((H_eq, np.transpose(cycle_rays)), axis=0)  # Add found linearities from H_ineq to H_eq

    # Remove duplicates from H_ineq and H_eq
    if redund_after_polco:
        H_ineq_original = H_ineq
        H_ineq_normalized = np.transpose(normalize_columns(np.transpose(H_ineq.astype(dtype='float')), verbose=verbose))
        # unique_inds = find_unique_inds(H_ineq_normalized, verbose=verbose, tol=1e-9)
        # H_ineq_float = H_ineq_normalized[unique_inds, :]
        # H_ineq_original = H_ineq_original[unique_inds, :]
        H_ineq_float, unique_inds = np.unique(H_ineq_normalized, axis=0, return_index=True)
        H_ineq_original = H_ineq_original[unique_inds, :]

        # H_ineq_float = unique(H_ineq_normalized)

        # Find out if rows have been thrown away, and if so, do that as well
        # unique_inds = find_remaining_rows(H_ineq_float, H_ineq_normalized, verbose=verbose)
        # H_ineq_original = H_ineq_original[unique_inds, :]

        if verbose:
            mp_print("Size of H_eq after communication step:", H_eq.shape[0], H_eq.shape[1])

        use_custom_redund = True  # If set to false, redundancy removal with redund from lrslib is used
        if use_custom_redund:
            mp_print('Using custom redundancy removal')
            t1 = time()
            nonred_inds_ineq, cycle_rays = drop_redundant_rays(np.transpose(H_ineq_float), rays_are_unique=True, linearities=False, normalised=True)
            mp_print("Custom redund took %f sec" % (time()-t1))

            H_ineq = H_ineq_original[nonred_inds_ineq, :]

            # t1 = time()
            # H_eq = independent_rows(H_eq)
            # mp_print("Removing dependent rows in H_eq took %f sec" % (time() - t1))
        else:
            mp_print('Using classical redundancy removal')
            t2 = time()
            H_ineq = redund(H_ineq)
            mp_print("Redund took %f sec" % (time() - t2))
            t2 = time()
            H_eq = redund(H_eq)
            mp_print("Redund took %f sec" % (time() - t2))

    print("Size of H_ineq after redund:", H_ineq.shape[0], H_ineq.shape[1])
    print("Size of H_eq after redund:", H_eq.shape[0], H_eq.shape[1])
    count_after_ineq = len(H_ineq)
    count_after_eq = len(H_eq)

    linearity_rays = np.ndarray(shape=(0, H_eq.shape[1]))
    if only_rays and len(in_out_indices) > 0:
        linearities = np.transpose(iterative_nullspace(np.append(H_eq, H_ineq, axis=0), verbose=verbose))
        if linearities.shape[0] > 0:
            if verbose:
                print('Appending linearities')
            linearity_rays = np.append(linearity_rays, linearities, axis=0)
            linearity_rays = np.append(linearity_rays, -linearities, axis=0)

            H_eq = np.append(H_eq, linearities, axis=0)

    if verbose:
        print('Removed %d rows from H in total' % (count_before_eq + count_before_ineq - count_after_eq - count_after_ineq))

    return H_eq, H_ineq, linearity_rays


def calc_C_extreme_rays(H_eq, H_ineq, polco, processes, jvm_mem, path2mplrs, verbose=True):
    """
    Calculate the extreme rays of the cone C represented by inequalities H_total, resulting in
    the elementary conversion modes of the input system.
    :param H_eq:
    :param H_ineq:
    :param polco:
    :param processes:
    :param jvm_mem:
    :param path2mplrs:
    :param verbose:
    :return:
    """
    if verbose:
        print('Calculating extreme rays C of inequalities system H_eq, H_ineq')

    rays = get_extreme_rays(H_eq if len(H_eq) else None, H_ineq, verbose=verbose, polco=polco, processes=processes,
                            jvm_mem=jvm_mem, path2mplrs=path2mplrs)
    return rays


def post_process_rays(G, rays, linearity_rays, external_metabolites, extended_external_metabolites,
                      in_out_metabolites, amount_metabolites, only_rays, verbose=True):
    """

    :param G:
    :param rays:
    :param linearity_rays:
    :param external_metabolites:
    :param extended_external_metabolites:
    :param in_out_metabolites:
    :param amount_metabolites:
    :param only_rays:
    :param verbose:
    :return:
    """
    # When calculating only extreme rays, we need to add linealities in both directions
    if only_rays and len(in_out_metabolites) > 0:
        rays = np.append(rays, linearity_rays, axis=0)

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays

    if verbose:
        print('Inflating rays')

    if only_rays:
        rays_inflated = inflate_matrix(rays, external_metabolites, amount_metabolites)
    else:
        rays_inflated = inflate_matrix(rays, extended_external_metabolites,
                                       amount_metabolites + len(in_out_metabolites))

    if verbose:
        print('Removing non-unique rays')

    # Merge bidirectional metabolites again, and drop duplicate rows
    if not only_rays:
        rays_inflated[:, in_out_metabolites] = np.subtract(rays_inflated[:, in_out_metabolites],
                                                           rays_inflated[:, G.shape[1]:])
    rays_merged = np.asarray(rays_inflated[:, :G.shape[1]], dtype='object')
    rays_unique = unique(rays_merged)
    # rays_unique = redund(rays_merged)

    if verbose:
        print('Enumerated %d rays' % len(rays_unique))

    return rays_unique


def get_conversion_cone(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                        only_rays=False, verbose=False, redund_after_polco=True, polco=False, processes=None, jvm_mem=None, path2mplrs=None):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).

    :param only_rays: return only the extreme rays of the conversion cone, and not the elementary vectors (ECMs instead of ECVs)
    :param verbose: print status messages during enumeration
    :param redund_after_polco: Optionally remove redundant rays from H_eq and H_ineq before final extreme ray enumeration by Polco
    :param polco: set to True to make computation with polco
    :param processes: integer value giving the number of processes
    :param jvm_mem: tuple of integer giving the minimum and maximum number of java VM memory in GB
    :param path2mplrs: absolute path to mplrs binary
    :return: matrix with conversion cone "c" as row vectors
    """
    calculate_linearities()
    calc_C0_dual_extreme_rays()
    calc_H()
    calc_C_extreme_rays()
    post_process_rays()


def get_pseudo_external_direction(network, metabolite_index):
    number_pos = 0
    number_neg = 0

    involved_reactions = np.where(network.N[metabolite_index, :] != 0)[0]

    for reaction_index in involved_reactions:
        reaction = network.reactions[reaction_index]
        if reaction.reversible:
            return 'both'

        stoich = network.N[metabolite_index, reaction_index]
        if stoich > 0:
            number_pos += 1
        elif stoich < 0:
            number_neg += 1

    return 'output' if number_neg == 0 else ('input' if number_pos == 0 else 'both')


def get_adjacent_metabolites(adjacency_matrix, metabolite_index):
    return np.where(adjacency_matrix[metabolite_index, :] != 0)[0]


def get_matrix_information(matrix):
    total_cells = matrix.shape[0] * matrix.shape[1]
    total_nonzero = np.count_nonzero(matrix)
    density = float(total_nonzero) / total_cells
    max = np.max(matrix)

    return density, max


def print_network_information(name, network, only_count=False):
    density, max = get_matrix_information(network.N)
    metabolites, reactions = network.N.shape
    number_external = len(network.external_metabolite_indices())
    print('%s: density %.2f, %d metabolites (%d ext), %d reactions, max %.2f' %
          (name, density, metabolites, number_external, reactions, max))

    if not only_count:
        print('metabolites: %s' % ', '.join(['%s (%s)' % (met.id, 'ext' if met.is_external else 'int') for met in network.metabolites]))


def replace_conversions_into_network(network, temp_network, conversions, active_reactions, verbose=False):
    if conversions.shape[0] == 0:
        print('The following metabolites have no viable conversion:',
              ', '.join([met.id for met in temp_network.metabolites]))
        return

    if verbose:
        print('Got %d conversions' % conversions.shape[0])

    if conversions.shape[0] > 400:
        print('Running redund on conversions')
        input("now at redund 547")
        conversions = redund(conversions)

    if verbose:
        print('Removing %d reactions used in conversions from network, and nullified %d metabolites' %
              (len(active_reactions),
               len(temp_network.metabolites) - len(temp_network.external_metabolite_indices())))

    keep_reactions = np.setdiff1d(range(len(network.reactions)), active_reactions)
    network.N = network.N[:, keep_reactions]
    network.reactions = list(np.asarray(network.reactions)[keep_reactions])

    if verbose:
        print('Adding %d uncompressed conversions to network' % conversions.shape[0])
    network.N = np.append(network.N, np.transpose(temp_network.uncompress(conversions)), axis=1)

    for _ in conversions:
        network.reactions.append(Reaction('conversion', 'conversion', reversible=False))

    if verbose:
        print('Running redund on conversions in full network stoichiometry')
    conversion_indices = np.where([r.id == 'conversion' for r in network.reactions])[0]
    natural_reaction_indices = np.setdiff1d(range(network.N.shape[1]), conversion_indices)
    all_conversions = network.N[:, conversion_indices]
    all_naturals = network.N[:, natural_reaction_indices]
    input("now at redund 574")
    reduced_conversions = np.transpose(redund(np.transpose(all_conversions)))
    network.N = np.append(all_naturals, reduced_conversions, axis=1)
    network.reactions = [r for index, r in enumerate(network.reactions) if index in natural_reaction_indices] + \
                        [Reaction('conversion', 'conversion', reversible=False) for _ in
                         range(reduced_conversions.shape[1])]
    return


def subnetwork(network, metabolite_indices, reaction_indices):
    temp_network = Network()
    temp_network.N = network.N[:, reaction_indices]
    temp_network.N = temp_network.N[metabolite_indices, :]
    # temp_network.N = np.transpose(redund(np.transpose(temp_network.N)))
    temp_network.reactions = list(np.asarray(network.reactions)[reaction_indices])
    temp_network.compressed = True
    temp_network.uncompressed_metabolite_ids = [met.id for met in network.metabolites]
    temp_network.metabolites = [Metabolite(network.metabolites[index].id, network.metabolites[index].name,
                                           network.metabolites[index].compartment,
                                           network.metabolites[index].is_external,
                                           network.metabolites[index].direction) for index in metabolite_indices]
    return temp_network


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
