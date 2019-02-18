from time import time

from .helpers import *
from .network import Network, Reaction, Metabolite
from .nullspace import iterative_nullspace


def normalize_rows(M):
    row_max = M.max(axis=1)
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


def get_clementine_conversion_cone(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                                   verbose=True):
    """
    Calculates the conversion cone using Superior Clementine Equality Intersection (all rights reserved).
    Follows the general Double Description method by Motzkin, using G as initial basis and intersecting
    hyperplanes of internal metabolites = 0.
    :param N:
    :param external_metabolites:
    :param reversible_reactions:
    :param input_metabolites:
    :param output_metabolites:
    :return:
    """
    amount_metabolites, amount_reactions = N.shape[0], N.shape[1]
    internal_metabolites = np.setdiff1d(range(amount_metabolites), external_metabolites)

    identity = np.identity(amount_metabolites)
    equalities = [identity[:, index] for index in internal_metabolites]

    # Compose G of the columns of N
    G = np.transpose(N)

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_reactions:
            G = np.append(G, [-G[reaction_index, :]], axis=0)

    # For each internal metabolite, intersect the intermediary cone with an equality to 0 for that metabolite
    for index, internal_metabolite in enumerate(internal_metabolites):
        if verbose:
            print('Iteration %d/%d' % (index, len(internal_metabolites)))

        # Find conversions that use this metabolite
        active_conversions = np.asarray([conversion_index for conversion_index in range(G.shape[0])
                              if G[conversion_index, internal_metabolite] != 0])

        # Skip internal metabolites that aren't used anywhere
        if len(active_conversions) == 0:
            if verbose:
                print('Skipping internal metabolite #%d, since it is not used by any reaction\n' % internal_metabolite)
            continue

        # Project conversions that use this metabolite onto the hyperplane internal_metabolite = 0
        projections = np.dot(G[active_conversions, :], equalities[index])
        positive = active_conversions[np.argwhere(projections > 0)[:, 0]]
        negative = active_conversions[np.argwhere(projections < 0)[:, 0]]
        candidates = np.ndarray(shape=(0, amount_metabolites))

        if verbose:
            print('Adding %d candidates' % (len(positive) * len(negative)))

        # Make convex combinations of all pairs (positive, negative) such that their internal_metabolite = 0
        for pos in positive:
            for neg in negative:
                candidate = np.add(G[pos, :], G[neg, :] * (G[pos, internal_metabolite] / -G[neg, internal_metabolite]))
                candidates = np.append(candidates, [candidate], axis=0)

        # Keep only rays that satisfy internal_metabolite = 0
        keep = np.setdiff1d(range(G.shape[0]), np.append(positive, negative, axis=0))
        if verbose:
            print('Removing %d rays\n' % (G.shape[0] - len(keep)))
        G = G[keep, :]
        G = np.append(G, candidates, axis=0)
        # G = drop_nonextreme(G, get_zero_set(G, equalities), verbose=verbose)
        G = redund(G, verbose=verbose)

    return G


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


def unique(matrix):
    unique_set = {tuple(row) for row in matrix if np.count_nonzero(row) > 0}
    return np.vstack(unique_set) if len(unique_set) else to_fractions(np.ndarray(shape=(0, matrix.shape[1])))


def get_conversion_cone(N, external_metabolites=[], reversible_reactions=[], input_metabolites=[], output_metabolites=[],
                        only_rays=False, symbolic=True, verbose=False):
    """
    Calculates the conversion cone as described in (Urbanczik, 2005).
    :param N: stoichiometry matrix
    :param external_metabolites: list of row numbers (0-based) of metabolites that are tagged as in/outputs
    :param reversible_reactions: list of booleans stating whether the reaction at this column is reversible
    :param input_metabolites: list of row numbers (0-based) of metabolites that are taggede as inputs
    :param output_metabolites: list of row numbers (0-based) of metabolites that are taggede as outputs
    :param only_rays: return only the extreme rays of the conversion cone, and not the elementary vectors (ECMs instead of ECVs)
    :param symbolic: use fractional numbers with symbolic algebra instead of floating point
    :param verbose: print status messages during enumeration
    :return: matrix with conversion cone "c" as row vectors
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

    # TODO: remove debug block
    # G = np.asarray(G * 10**3, dtype=np.int64)

    G_exp = G[:,:]
    G_rev = np.ndarray(shape=(0, G.shape[1]), dtype='object')
    G_irrev = np.ndarray(shape=(0, G.shape[1]), dtype='object')

    # Add reversible reactions (columns) of N to G in the negative direction as well
    for reaction_index in range(G.shape[0]):
        if reaction_index in reversible_reactions:
            G_exp = np.append(G_exp, [-G[reaction_index, :]], axis=0)
            G_rev = np.append(G_rev, [-G[reaction_index, :]], axis=0)
        else:
            G_irrev = np.append(G_irrev, [G[reaction_index, :]], axis=0)


    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
         print('Calculating null space of inequalities system G')
    # linearities = np.transpose(nullspace_polco(G, verbose=verbose))
    linearities = np.transpose(iterative_nullspace(G, verbose=verbose))
    # linearities = to_fractions(np.transpose(nullspace(np.asarray(G, dtype='float64'), symbolic=False)))
    # linearities = np.transpose(nullspace_terzer(G, verbose=verbose))
    # linearities = np.loadtxt("/tmp/lin_ecoli2.csv", delimiter=',', dtype='int')
    if linearities.shape[0] == 0:
        linearities = np.ndarray(shape=(0, G.shape[1]))

    # if symbolic and linearities.shape[0] > 0:
    #     assert np.sum(np.sum(np.dot(G, np.transpose(linearities)), axis=0)) == 0
    # elif linearities.shape[0] > 0:
    #     sum = np.sum(np.sum(np.dot(G, np.transpose(linearities)), axis=0))
    #     assert -10 ** -6 <= sum <= 10 ** -6
    linearities_deflated = deflate_matrix(linearities, external_metabolites)

    # Calculate H as the union of our linearities and the extreme rays of matrix G (all as row vectors)
    if verbose:
         print('Calculating extreme rays H of inequalities system G')

    # Calculate generating set of the dual of our initial conversion cone C0, C0*
    rays = get_extreme_rays(np.append(linearities, G_rev, axis=0), G_irrev, verbose=verbose, symbolic=symbolic)

    if rays.shape[0] == 0:
        print('Warning: given system has no nonzero inequalities H. Returning empty conversion cone.')
        return to_fractions(np.ndarray(shape=(0, G.shape[1])))

    if verbose:
        print('Deflating H')
    rays_deflated = deflate_matrix(rays, external_metabolites)

    if verbose:
        print('Expanding H with metabolite direction constraints')
    # Add bidirectional (in- and output) metabolites in reverse direction
    rays_split = split_columns(rays_deflated, in_out_indices) if not only_rays else rays_deflated
    linearities_split = split_columns(linearities_deflated, in_out_indices) if not only_rays else linearities_deflated

    H_ineq = rays_split
    H_eq = linearities_split

    # Add input/output constraints to H_ineq
    if not H_ineq.shape[0]:
        H_ineq = np.zeros(shape=(1, H_ineq.shape[1]))

    identity = to_fractions(np.identity(H_ineq.shape[1]))

    # Bidirectional (in- and output) metabolites
    # for list_index, inout_metabolite_index in enumerate(in_out_indices):
    #     index = inout_metabolite_index
    #     H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)
    #
    #     if not only_rays:
    #         index = len(external_metabolites) + list_index
    #         H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

    # Inputs
    for input_metabolite in input_metabolites:
        index = external_metabolites.index(input_metabolite)
        H_ineq = np.append(H_ineq, [-identity[index, :]], axis=0)

    # Outputs
    for output_metabolite in output_metabolites:
        index = external_metabolites.index(output_metabolite)
        H_ineq = np.append(H_ineq, [identity[index, :]], axis=0)

    if verbose:
        print('Reducing rows in H with redund')

    count_before_ineq = len(H_ineq)
    H_ineq = redund(H_ineq)
    count_after_ineq = len(H_ineq)

    count_before_eq = len(H_eq)
    H_eq = redund(H_eq)
    count_after_eq = len(H_eq)

    if verbose:
        print('Removed %d rows from H' % (count_before_eq + count_before_ineq - count_after_eq - count_after_ineq))

    # Calculate the extreme rays of the cone C represented by inequalities H_total, resulting in
    # the elementary conversion modes of the input system.
    if verbose:
        print('Calculating extreme rays C of inequalities system H_eq, H_ineq')

    # rays = np.asarray(list(get_extreme_rays_efmtool(H_total)))
    # rays = np.asarray(list(get_extreme_rays(None, H_total, verbose=verbose)))
    # rays = np.asarray(list(get_extreme_rays(H_eq if len(H_eq) else None, H_ineq, verbose=verbose, symbolic=symbolic)))
    # rays = get_extreme_rays_cdd(H_total)

    rays = get_extreme_rays(H_eq if len(H_eq) else None, H_ineq, verbose=verbose, symbolic=symbolic)

    if rays.shape[0] == 0:
        print('Warning: no feasible Elementary Conversion Modes found')
        return rays

    if only_rays and len(in_out_metabolites) > 0:
        linearities = np.transpose(iterative_nullspace(np.append(H_eq, H_ineq, axis=0), verbose=verbose))
        if linearities.shape[0] > 0:
            if verbose:
                print('Appending linearities')
            rays = np.append(rays, linearities, axis=0)

    if verbose:
        print('Inflating rays')

    if only_rays:
        rays_inflated = inflate_matrix(rays, external_metabolites, amount_metabolites)
    else:
        rays_inflated = inflate_matrix(rays, extended_external_metabolites, amount_metabolites + len(in_out_metabolites))

    if verbose:
        print('Removing non-unique rays')


    # Merge bidirectional metabolites again, and drop duplicate rows
    # np.unique() requires non-object matrices, so here we cast our results into float64.
    if not only_rays:
        rays_inflated[:, in_out_metabolites] = np.subtract(rays_inflated[:, in_out_metabolites], rays_inflated[:, G.shape[1]:])
    rays_merged = np.asarray(rays_inflated[:, :G.shape[1]], dtype='object')
    rays_unique = unique(rays_merged)
    # rays_unique = redund(rays_merged)

    if verbose:
        print('Enumerated %d rays' % len(rays_unique))

    return rays_unique


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


def iterative_conversion_cone(network, max_metabolites=30, verbose=True):

    if verbose:
        print_network_information('N', network)

    # Hide biomass function
    network.remove_objective_reaction()

    adjacency = get_metabolite_adjacency(network.N)
    metabolite_indices = range(len(network.metabolites))
    internal_indices = [index for index,met in enumerate(network.metabolites) if not met.is_external]
    original_length = len(internal_indices)

    total_conversions_added = 0

    def get_connectivity(metabolite_index):
        return np.sum(adjacency[metabolite_index, :])

    # Perform one round of conversion cone calculation
    while len(internal_indices) > 0:
        print('\n======== Did %d/%d intermediary metabolites ========\n' % ((original_length - len(internal_indices)), original_length))

        # Sort by adjacency, descending (because of pop())
        internal_indices.sort(key=get_connectivity, reverse=True)

        initial = internal_indices.pop()
        connectivity = get_connectivity(initial)
        if connectivity > max_metabolites:
            print('No more internal metabolites below connectivity threshold')
            break
        selection = [initial]
        last_round = [initial]
        all_active_reactions = []
        bordering = list(get_adjacent_metabolites(adjacency, initial))

        print('Adding initial metabolite %s with %d adjacent metabolites' %
              (network.metabolites[initial].id, connectivity))

        while len(selection) + len(bordering) <= max_metabolites:
            current_round = []
            for metabolite_index in last_round:
                active_reactions = np.where(network.N[metabolite_index, :] != 0)[0]
                all_active_reactions.extend(active_reactions)
                all_active_reactions = list(np.unique(all_active_reactions))
                adjacent = get_adjacent_metabolites(adjacency, metabolite_index)
                adjacent = list(np.setdiff1d(adjacent, selection))
                adjacent = list(np.setdiff1d(adjacent, network.external_metabolite_indices()))
                adjacent.sort(key=get_connectivity)

                for adjacent_index in adjacent:
                    inner_adjacent = get_adjacent_metabolites(adjacency, adjacent_index)
                    inner_adjacent = list(np.setdiff1d(inner_adjacent, np.union1d(selection, bordering)))
                    connectivity = len(inner_adjacent)
                    total = len(selection) + len(bordering) + connectivity
                    if total <= max_metabolites:
                        current_round.append(adjacent_index)
                        selection.append(adjacent_index)
                        internal_indices = list(np.setdiff1d(internal_indices, selection))
                        bordering.extend(inner_adjacent)
                        bordering = list(np.unique(np.setdiff1d(bordering, selection)))
                        active_reactions = np.where(network.N[adjacent_index, :] != 0)[0]
                        all_active_reactions.extend(active_reactions)
                        all_active_reactions = list(np.unique(all_active_reactions))
                        print('Added metabolite %s with %d adjacent metabolites (total: %d)' %
                              (network.metabolites[adjacent_index].id, connectivity, len(selection) + len(bordering)))
                    else:
                        print('Skipping metabolite %s with %d adjacent metabolites' % (network.metabolites[adjacent_index].id, connectivity))


            last_round = current_round

            if len(current_round) == 0:
                break

        # Check if there are any reactions active with chosen metabolites
        if len(all_active_reactions) == 0:
            print('The following metabolites have no viable conversion:',
                  ', '.join([network.metabolites[id].id for id in selection]))
            continue

        print ('Info: using %d/%d adjacent metabolites (%d incl bordering)' % (len(selection), max_metabolites, len(selection) + len(bordering)))

        # Add bordering metabolites to our subsystem
        selection.extend(bordering)
        selection = list(np.unique(selection))

        # Create new network with only selected metabolites and their reactions
        temp_network = subnetwork(network, selection, all_active_reactions)

        # Mark all non-external non-bordering metabolites as internal
        for deflated_index, inflated_index in enumerate(selection):
            if inflated_index in bordering or network.metabolites[inflated_index].is_external:
                temp_network.metabolites[deflated_index].is_external = True
            else:
                temp_network.metabolites[deflated_index].is_external = False

            temp_network.metabolites[deflated_index].direction = get_pseudo_external_direction(temp_network, deflated_index)

        if verbose:
            print_network_information('T', temp_network)

        conversions = get_conversion_cone(temp_network.N, temp_network.external_metabolite_indices(),
                                          temp_network.reversible_reaction_indices(),
                                          temp_network.input_metabolite_indices(),
                                          temp_network.output_metabolite_indices(),
                                          only_rays=True, verbose=verbose, symbolic=True)

        add_conversions_to_network(network, temp_network, conversions, all_active_reactions, verbose=verbose)

        if verbose:
            print_network_information('N', network, only_count=True)
            print('Done with modifying network reactions, recalculating adjacency')


        # total_conversions_added += conversions.shape[0]
        # if total_conversions_added > 200:
        #     total_conversions_added = 0
        #     if verbose:
        #         print('Running redund on full stoichiometry')
        #     network.N = np.transpose(redund(np.transpose(network.N), verbose=verbose))
        #     network.reactions = [Reaction('conversion', 'conversion', reversible=False) for _ in range(network.N.shape[1])]

        # TODO: maybe remove now superfluous internal metabolites as well

        adjacency = get_metabolite_adjacency(network.N)

    # Compress and restore the biomass reaction in the network
    # network.compress(verbose=verbose)
    network.restore_objective_reaction()

    if verbose:
        print('Calculating biomass conversions')

    # Calculate conversions to biomass
    iterative_biomass_conversions(network, verbose=verbose)

    if verbose:
        print('Calculating any remaining conversions')

    # TODO: another conversion enumeration is only necessary if there are remaining internal metabolites,
    # e.g. when there were internal ones with high connectivity

    # Calculate conversions left by internal metabolites skipped because of the connectivity threshold above
    conversion_cone = get_conversion_cone(network.N, network.external_metabolite_indices(),
                                          network.reversible_reaction_indices(),
                                          network.input_metabolite_indices(),
                                          network.output_metabolite_indices(),
                                          verbose=verbose, symbolic=True)
    return network.uncompress(conversion_cone)


def add_conversions_to_network(network, temp_network, conversions, active_reactions, verbose=False):
    if conversions.shape[0] == 0:
        print('The following metabolites have no viable conversion:',
              ', '.join([met.id for met in temp_network.metabolites]))
        return

    if verbose:
        print('Got %d conversions' % conversions.shape[0])

    if conversions.shape[0] > 400:
        print('Running redund on conversions')
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
    temp_network.N = np.transpose(redund(np.transpose(temp_network.N)))
    temp_network.reactions = list(np.asarray(network.reactions)[reaction_indices])
    temp_network.compressed = True
    temp_network.uncompressed_metabolite_ids = [met.id for met in network.metabolites]
    temp_network.metabolites = [Metabolite(network.metabolites[index].id, network.metabolites[index].name,
                                           network.metabolites[index].compartment,
                                           network.metabolites[index].is_external,
                                           network.metabolites[index].direction) for index in metabolite_indices]
    return temp_network


def subnetwork_with_metabolites(network, metabolite_indices):
    active_reactions = []
    selected_metabolites = []
    adjacency = get_metabolite_adjacency(network.N)

    for index in metabolite_indices:
        reactions = list(np.where(network.N[index, :] != 0)[0])
        active_reactions = list(np.unique(active_reactions + reactions))

        adjacents = list(get_adjacent_metabolites(adjacency, index))
        selected_metabolites = list(np.unique(selected_metabolites + [index] + adjacents))

    return subnetwork(network, selected_metabolites, active_reactions)


def iterative_biomass_conversions(network, verbose=False):
    """
    Turns X1 + X2 + X3 + X4 > biomass + X9 + X10 into
    X1 + X2 > A
    A + X3 > B
    B + X4 > C
    C > biomass + X9 + X10

    This reduces the problem size of enumerating conversions.
    Has to be called directly after network.restore_objective_reaction().

    :param network:
    :return:
    """
    substrate_indices, product_indices = network.get_objective_reagents()

    biomass_stoich = network.N[:, -1]
    orig_shape = network.N.shape

    # Hide biomass reaction again
    # network.N = network.N[:, :-1]
    network.drop_reactions([network.N.shape[1] - 1])

    # Add virtual metabolites
    zeroes = to_fractions(np.zeros(shape=(1, 1)))
    number_virtual_metabolites = len(substrate_indices) - 1
    virtual_metabolite_indices = [i + orig_shape[0] for i in range(number_virtual_metabolites)]
    virtual_stoich_metabolites = np.repeat(np.repeat(zeroes, number_virtual_metabolites, axis=0), network.N.shape[1], axis=1)
    network.N = np.append(network.N, virtual_stoich_metabolites, axis=0)
    network.metabolites.extend([Metabolite('virtual %d' % i, 'virtual %d' % i, 'e', True) for i in range(1, number_virtual_metabolites + 1)])

    initial = (substrate_indices[0], substrate_indices[1])
    pairs = [initial]
    stoichs = [(biomass_stoich[substrate_indices[0]], biomass_stoich[substrate_indices[1]])]

    for i in range(number_virtual_metabolites - 1):
        substrate = substrate_indices[i+2]
        stoichiometry = biomass_stoich[substrate]
        virtual_metabolite = virtual_metabolite_indices[i]
        pairs.append((virtual_metabolite, substrate))
        stoichs.append((Fraction(-1), stoichiometry))

    # Compute conversions up to and including the last virtual metabolite (C in the function description)
    for i, pair in enumerate(pairs):
        print('\n======== Eliminating biomass substrate %d/%d ========\n' % (i+1, len(pairs)))
        # Mark substrates as internal
        network.metabolites[pair[0]].is_external = False
        network.metabolites[pair[1]].is_external = False

        # Add virtual reaction
        column = np.repeat(zeroes, network.N.shape[0], axis=0)
        column[pair[0]] = stoichs[i][0]
        column[pair[1]] = stoichs[i][1]
        column[virtual_metabolite_indices[i]] = Fraction(1)
        network.N = np.append(network.N, column, axis=1)
        network.reactions.extend([Reaction('virtual interim', 'virtual interim', False)])

        # Create subnetwork with pair in it
        active_reactions = list(np.unique(list(np.where(network.N[pair[0], :] != 0)[0]) + list(np.where(network.N[pair[1], :] != 0)[0])))
        adjacency = get_metabolite_adjacency(network.N)
        selection = list(np.unique(list(get_adjacent_metabolites(adjacency, pair[0])) + list(get_adjacent_metabolites(adjacency, pair[1])) + [pair[0], pair[1]]))
        temp_network = subnetwork(network, selection, active_reactions)

        # Mark all other substrates as external
        other_ids = [network.metabolites[index].id for index in selection if index not in pair]
        for metabolite in temp_network.metabolites:
            if metabolite.id in other_ids:
                metabolite.is_external = True

        if verbose:
            print_network_information('T', temp_network)

        # Calculate conversions
        conversions = get_conversion_cone(temp_network.N, temp_network.external_metabolite_indices(),
                                              temp_network.reversible_reaction_indices(),
                                              temp_network.input_metabolite_indices(),
                                              temp_network.output_metabolite_indices(),
                                              only_rays=True, verbose=verbose, symbolic=True)
        add_conversions_to_network(network, temp_network, conversions, active_reactions, verbose=verbose)

        if verbose:
            print_network_information('N', network, only_count=True)

    # Add virtual reaction
    column = np.repeat(zeroes, network.N.shape[0], axis=0)
    column[virtual_metabolite_indices[i]] = Fraction(-1)
    for index in product_indices:
        column[index] = biomass_stoich[index]
    network.N = np.append(network.N, column, axis=1)
    network.reactions.extend([Reaction('objective', 'Objective reaction', False)])

    # Mark last virtual metabolites as internal
    network.metabolites[-1].is_external = False

    # Find all adjacents to the last virtual metabolite
    last_metabolite_index = virtual_metabolite_indices[-1]
    last_metabolite_id = network.metabolites[last_metabolite_index].id
    adjacency = get_metabolite_adjacency(network.N)
    adjacents = list(get_adjacent_metabolites(adjacency, last_metabolite_index))

    # Collect all product indices, plus the index of the last virtual metabolite
    relevant_metabolites = adjacents + [last_metabolite_index]
    active_reactions = list(np.where(network.N[last_metabolite_index, :] != 0)[0])
    temp_network = subnetwork(network, relevant_metabolites, active_reactions)

    # Mark all adjacents to last virtual metabolite as external
    for metabolite in temp_network.metabolites:
        if metabolite.id != last_metabolite_id:
            metabolite.is_external = True

    if verbose:
        print('Computing final conversion to biomass')

    # Compute final conversion to biomass
    conversions = get_conversion_cone(temp_network.N, temp_network.external_metabolite_indices(),
                            temp_network.reversible_reaction_indices(),
                            temp_network.input_metabolite_indices(),
                            temp_network.output_metabolite_indices(),
                            only_rays=True, verbose=verbose, symbolic=True)
    add_conversions_to_network(network, temp_network, conversions, active_reactions, verbose=verbose)

    # Set products to their original status
    for index in product_indices:
        metabolite = network.metabolites[index]
        metabolite.is_external = metabolite.orig_external
        metabolite.direction = metabolite.orig_direction

    # Remove virtual metabolites
    network.drop_metabolites(virtual_metabolite_indices)

    return



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
