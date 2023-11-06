from time import time

from .helpers import *
from .network import Network, Reaction, Metabolite
from .nullspace import iterative_nullspace
from .custom_redund import drop_redundant_rays, remove_cycles_redund
from .intersect_directly_mpi import independent_rows
from ecmtool import mpi_wrapper


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


def split_columns(matrix, columns):
    matrix = np.append(matrix, -matrix[:, columns], axis=1)
    return matrix


def get_conversion_cone(N, external_metabolites, reversible_reactions, input_metabolites, output_metabolites,
                        verbose=True):
    """
    Main function to calculate ecms in one go, i.e. without using the modular steps as used in main.py. Currently,
    this function does not allow certain options such as only_rays, hiding-metabolites, etc. Please look at the
    work flow in main.py to see how you can use these options.
    Parameters
    ----------
    N: stoichiometric matrix (rows are metabolites, columns are reactions)
    external_metabolites: index list of metabolites that are external
    reversible_reactions: index list of reactions that are reversible.
    input_metabolites: index list of external metabolites that can only be used as an input
    output_metabolites: index list of external metabolites that can only be used as an output
    verbose

    Returns
    -------
    cone: Conversion cone. Matrix with ECMs as columns, external metabolites as rows.
    """
    print("Calculating linearities in original network.")
    linearity_data = calculate_linearities(N, reversible_reactions, external_metabolites, input_metabolites,
                                           output_metabolites, verbose)

    linearities, linearities_deflated, G_rev, G_irrev, amount_metabolites, \
    extended_external_metabolites, in_out_indices = linearity_data

    C0_dual_rays = calc_C0_dual_extreme_rays(linearities, G_rev, G_irrev, polco=True, processes=1, jvm_mem=None,
                                             path2mplrs=None)

    print("Calculating H. Adding steady-state, irreversibility constraints, then discarding redundant inequalities.")
    H_eq, H_ineq, linearity_rays = calc_H(rays=C0_dual_rays, linearities_deflated=linearities_deflated,
                                          external_metabolites=external_metabolites,
                                          input_metabolites=input_metabolites,
                                          output_metabolites=output_metabolites,
                                          in_out_indices=in_out_indices,
                                          redund_after_polco=True, only_rays=False, verbose=True)

    """Calculate extreme rays of conversion cone."""
    print("Calculating extreme rays of final cone.")
    cone = calc_C_extreme_rays(H_eq, H_ineq, polco=True, processes=1, jvm_mem=None, path2mplrs=None)
    return cone


def calculate_linearities(N, reversible_reactions, external_metabolites, input_metabolites, output_metabolites,
                          verbose=False):
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


def calc_H(rays=None, linearities_deflated=None, external_metabolites=None, input_metabolites=None,
           output_metabolites=None, in_out_indices=None, redund_after_polco=True, only_rays=False, verbose=False):
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
    if mpi_wrapper.is_first_process():
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
        linearities_split = split_columns(linearities_deflated,
                                          in_out_indices) if not only_rays else linearities_deflated

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
        if mpi_wrapper.is_first_process():
            H_ineq_original = H_ineq
            H_ineq_normalized = np.transpose(
                normalize_columns(np.transpose(H_ineq.astype(dtype='float')), verbose=verbose))
            H_ineq_float, unique_inds = np.unique(H_ineq_normalized, axis=0, return_index=True)
            H_ineq_original = H_ineq_original[unique_inds, :]

        use_custom_redund = True  # If set to false, redundancy removal with redund from lrslib is used
        if use_custom_redund:
            mp_print('Using custom redundancy removal')
            if not mpi_wrapper.is_first_process():
                H_ineq_float = None
            H_ineq_float = mpi_wrapper.bcast(H_ineq_float, root=0)
            t1 = time()
            nonred_inds_ineq, cycle_rays = drop_redundant_rays(np.transpose(H_ineq_float), rays_are_unique=True,
                                                               linearities=False, normalised=True)
            if not mpi_wrapper.is_first_process():
                return None
            mp_print("Custom redund took %f sec" % (time() - t1))
            H_ineq = H_ineq_original[nonred_inds_ineq, :]
        else:
            if mpi_wrapper.is_first_process():
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
        print('Removed %d rows from H in total' % (
                count_before_eq + count_before_ineq - count_after_eq - count_after_ineq))

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

    # Merge bidirectional metabolites again, and drop duplicate rows
    if not only_rays:
        rays_inflated[:, in_out_metabolites] = np.subtract(rays_inflated[:, in_out_metabolites],
                                                           rays_inflated[:, G.shape[1]:])
    rays_merged = np.asarray(rays_inflated[:, :G.shape[1]], dtype='object')

    if verbose:
        print('Enumerated %d rays' % len(rays_merged))

    return rays_merged
