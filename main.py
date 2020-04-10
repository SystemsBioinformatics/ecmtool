import os, sys

import numpy as np
from time import time
from scipy.optimize import linprog
from argparse import ArgumentParser, ArgumentTypeError
from sklearn.preprocessing import normalize


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


with HiddenPrints():
    from ecmtool.helpers import get_efms, get_metabolite_adjacency, redund, to_fractions
    from ecmtool.intersect_directly_mpi import intersect_directly, print_ecms_direct, remove_cycles, \
        compress_after_cycle_removing, normalize_columns, check_if_intermediate_cone_exists
    from ecmtool.helpers import mp_print
    from ecmtool.network import extract_sbml_stoichiometry
    from ecmtool.conversion_cone import get_conversion_cone, iterative_conversion_cone, unique
    from ecmtool.functions_for_Erik import check_bijection_Erik


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def ecm_satisfies_stoichiometry(stoichiometry, ecm):
    allowed_error = 10 ** -6
    solution = linprog(c=[0] * stoichiometry.shape[1], A_eq=stoichiometry, b_eq=ecm,
                       bounds=[(-1000, 1000)] * stoichiometry.shape[1], options={'tol': allowed_error})
    return solution.status == 0


def print_ECMs(cone, debug_tags, network, orig_N, add_objective_metabolite, check_feasibility):
    for index, ecm in enumerate(cone):
        # Normalise by objective metabolite, if applicable
        objective_index = -1 - len(debug_tags)
        objective = ecm[objective_index]
        if add_objective_metabolite and objective > 0:
            ecm /= objective

        metabolite_ids = [met.id for met in
                          network.metabolites] if not network.compressed else network.uncompressed_metabolite_ids

        print('\nECM #%d:' % (index + 1))
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%s\t\t->\t%.4f' % (metabolite_ids[metabolite_index], stoichiometry_val))

        if check_feasibility:
            satisfied = ecm_satisfies_stoichiometry(orig_N, cone[index, :])
            print('ECM satisfies stoichiometry' if satisfied else 'ECM does not satisfy stoichiometry')


def remove_close_vectors(matrix, threshold=10 ** -6, verbose=True):
    i = 0
    new_matrix = matrix

    if verbose:
        print('Removing vectors with small distance to others')

    while i < new_matrix.shape[0]:
        temp_matrix = new_matrix
        unique_indices = range(i + 1) + [index + i + 1 for index in
                                         find_matching_vector_indices(temp_matrix[i, :], temp_matrix[i + 1:, :],
                                                                      near=False, threshold=threshold)]

        if verbose:
            print('%.2f%% (removed %d/%d)' % (
                100 * float(i) / new_matrix.shape[0], matrix.shape[0] - new_matrix.shape[0], matrix.shape[0]))

        new_matrix = temp_matrix[unique_indices, :]
        i += 1

    return new_matrix


def find_matching_vector_indices(vector, matrix, near=True, threshold=10 ** -6, verbose=True):
    """
    Returns indices of row vectors with small distance to vector if near==True, or with large distance otherwise.
    :param vector:
    :param matrix:
    :param near:
    :param threshold:
    :param verbose:
    :return:
    """
    matching_indices = []
    for i in range(matrix.shape[0]):
        other_vector = matrix[i, :]
        dist = np.linalg.norm(vector - other_vector)
        if (near and dist < threshold) or (not near and dist > threshold):
            matching_indices.append(i)
    return matching_indices


def vectors_in_cone(vector_matrix, cone_matrix, network, verbose=True):
    cone_trans = cone_matrix.transpose()
    in_cone = True
    for index, vector in enumerate(vector_matrix):
        if verbose:
            print('Checking %d/%d' % (index + 1, len(vector_matrix)))

        allowed_error = 10 ** -6
        solution = linprog(c=[1] * cone_trans.shape[1], A_eq=cone_trans, b_eq=vector,
                           bounds=[(0, 1000)] * cone_trans.shape[1], options={'tol': allowed_error})
        if solution.status != 0:
            print('EFM conversion does not fit inside conversion cone:')
            for metabolite_index, stoichiometry_val in enumerate(vector):
                if stoichiometry_val != 0.0:
                    print('%d %s\t\t->\t%.4f' % (
                        metabolite_index, network.metabolites[metabolite_index].id, stoichiometry_val))
            in_cone = False
        elif verbose:
            print('Support: ' + str([index for index, _ in enumerate(solution['x']) if abs(solution['x'][index]) > 1e-6]))

    return in_cone


# def vectors_in_cone(vector_matrix, cone_matrix, network, verbose=True):
#     cone_trans = cone_matrix.transpose()
#     in_cone = True
#     for index, vector in enumerate(vector_matrix):
#         if verbose:
#             print('Checking %d/%d' % (index+1, len(vector_matrix)))
#
#         matches = find_matching_vector_indices(vector, cone_matrix, near=True)
#         if len(matches) != 0:
#             print('EFM conversion does not fit inside conversion cone:')
#             for metabolite_index, stoichiometry_val in enumerate(vector):
#                 if stoichiometry_val != 0.0:
#                     print('%d %s\t\t->\t%.4f' % (
#                     metabolite_index, network.metabolites[metabolite_index].id, stoichiometry_val))
#             in_cone = False
#         elif verbose:
#             print('Support: ' + str([index for index,_ in enumerate(solution['x']) if abs(solution['x'][index]) > 1e-6]))
#
#     return in_cone


def check_bijection(conversion_cone, network, model_path, args, verbose=True):
    full_model = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                            determine_inputs_outputs=args.auto_direction,
                                            skip_external_reactions=True,
                                            external_compartment=args.external_compartment)
    if args.check_bijection:
        set_inoutputs(args.inputs, args.outputs, full_model)

    ex_N = full_model.N
    identity = np.identity(len(full_model.metabolites))
    reversibilities = [reaction.reversible for reaction in full_model.reactions]

    # Add exchange reactions so efmtool can calculate EFMs in steady state
    n_exchanges = 0
    for index, metabolite in enumerate(full_model.metabolites):
        if metabolite.is_external:
            reaction = identity[:, index] if metabolite.direction != 'output' else -identity[index]
            ex_N = np.append(ex_N, np.transpose([reaction]), axis=1)
            reversibilities.append(True if metabolite.direction == 'both' else False)
            n_exchanges += 1

    efms = get_efms(ex_N, reversibilities)
    if verbose:
        print('Calculating ECMs from EFMs')

    # Remove exchange reactions again from EFMs
    efms = efms[:, :-n_exchanges]
    efm_ecms = np.transpose(np.dot(np.asarray(full_model.N, dtype='object'), np.transpose(efms)))
    if verbose:
        print('Removing non-unique ECMs')
    efm_ecms_rounded = efm_ecms.round(decimals=6)
    efm_ecms_normalised = normalize(efm_ecms_rounded, axis=1)
    efm_ecms_unique = unique(efm_ecms_normalised)
    # efm_ecms_unique = redund(efm_ecms_unique)

    ecmtool_ecms_normalised = normalize(conversion_cone, axis=1)

    if verbose:
        print('Found %d efmtool-calculated ECMs, and %d ecmtool ones' % (
            len(efm_ecms_unique), len(ecmtool_ecms_normalised)))

    is_bijection = True

    # if verbose:
    #     print('Checking if efmtool ECMs satisfy stoichiometry')
    #
    # for index, ecm in enumerate(efm_ecms_unique):
    #     if verbose:
    #         print('Checking %d/%d' % (index+1, len(efm_ecms_unique)))
    #     if not ecm_satisfies_stoichiometry(full_model.N, ecm):
    #         print('efmtool ECM does not satisfy stoichiometry')

    if verbose:
        print('Checking bijection')

    if not vectors_in_cone(efm_ecms_unique, ecmtool_ecms_normalised, full_model, verbose):
        print('Calculated ECMs do not agree with EFM conversions')

    for index, ecm in enumerate(efm_ecms_unique):
        if verbose:
            print('Checking %d/%d (round 1/2)' % (index + 1, len(efm_ecms_unique)))
        close_vectors = find_matching_vector_indices(ecm, ecmtool_ecms_normalised, near=True)
        if len(close_vectors) != 1:
            is_bijection = False
            print(
                '\nCalculated ECM #%d not uniquely in enumerated list (got %d matches):' % (index, len(close_vectors)))
            for metabolite_index, stoichiometry_val in enumerate(ecm):
                if stoichiometry_val != 0.0:
                    print('%d %s\t\t->\t%.4f' % (
                        metabolite_index, network.uncompressed_metabolite_ids[metabolite_index], stoichiometry_val))

    for index, ecm in enumerate(ecmtool_ecms_normalised):
        if verbose:
            print('Checking %d/%d (round 2/2)' % (index + 1, len(ecmtool_ecms_normalised)))
        if ecm not in efm_ecms_unique:
            is_bijection = False
            print('\nEnumerated ECM #%d not in calculated list:' % index)
            for metabolite_index, stoichiometry_val in enumerate(ecm):
                if stoichiometry_val != 0.0:
                    print('%d %s\t\t->\t%.4f' % (
                        metabolite_index, network.uncompressed_metabolites_names[metabolite_index], stoichiometry_val))

    print('Enumerated ECMs and calculated ECMs are%s bijective' % ('' if is_bijection else ' not'))


def set_inoutputs(inputs, outputs, network):
    inputs = [int(index) for index in inputs.split(',') if len(index)]
    outputs = [int(index) for index in outputs.split(',') if len(index)]
    if len(outputs) < 1 and len(inputs) >= 1:
        # If no outputs are given, define all external metabolites that are not inputs as outputs
        mp_print('No output metabolites given or determined from model. All non-input metabolites will be defined as outputs.')
        outputs = np.setdiff1d(network.external_metabolite_indices(), inputs)

    network.set_inputs(inputs)
    network.set_outputs(outputs)
    if len(np.intersect1d(inputs, outputs)):
        for ind in np.intersect1d(inputs, outputs):
            mp_print(
                'Metabolite %s was marked as both only input and only output, which is impossible. It is set to both, for now.' % (
                    network.metabolites[ind].id))
        network.set_both(np.intersect1d(inputs, outputs))
    return


if __name__ == '__main__':
    start = time()

    parser = ArgumentParser(
        description='Calculate Elementary Conversion Modes from an SBML model. For medium-to large networks, be sure to define --inputs and --outputs. This reduces the enumeration problem complexity considerably.')
    parser.add_argument('--model_path', type=str, default='models/active_subnetwork_KO_5.xml',
                        help='Relative or absolute path to an SBML model .xml file')
    parser.add_argument('--direct', type=str2bool, default=True, help='Enable to intersect with equalities directly')
    parser.add_argument('--compress', type=str2bool, default=True,
                        help='Perform compression to which the conversions are invariant, and reduce the network size considerably (default: True)')
    parser.add_argument('--out_path', default='conversion_cone.csv',
                        help='Relative or absolute path to the .csv file you want to save the calculated conversions to (default: conversion_cone.csv)')
    parser.add_argument('--add_objective_metabolite', type=str2bool, default=True,
                        help='Add a virtual metabolite containing the stoichiometry of the objective function of the model (default: true)')
    parser.add_argument('--check_feasibility', type=str2bool, default=False,
                        help='For each found ECM, verify that a feasible flux exists that produces it (default: false)')
    parser.add_argument('--check_bijection', type=str2bool, default=False,
                        help='Verify completeness of found ECMs by calculating ECMs from EFMs and proving bijection (don\'t use on large networks) (default: false)')
    parser.add_argument('--print_metabolites', type=str2bool, default=True,
                        help='Print the names and IDs of metabolites in the (compressed) metabolic network (default: true)')
    parser.add_argument('--print_reactions', type=str2bool, default=False,
                        help='Print the names and IDs of reactions in the (compressed) metabolic network (default: true)')
    parser.add_argument('--print_conversions', type=str2bool, default=True,
                        help='Print the calculated conversion modes (default: true)')
    parser.add_argument('--external_compartment', type=str, default='e',
                        help='String indicating how the external compartment in metabolite_ids of SBML-file is marked. Please check if external compartment detection works by checking metabolite information before compression and with --primt metabolites true')
    parser.add_argument('--auto_direction', type=str2bool, default=True,
                        help='Automatically determine external metabolites that can only be consumed or produced (default: true)')
    parser.add_argument('--inputs', type=str, default='',
                        help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that can only be consumed')
    parser.add_argument('--outputs', type=str, default='',
                        help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that can only be produced. '
                             'If inputs are given, but no outputs, then everything not marked as input is marked as output.'
                             'If inputs and outputs are given, the possible remainder of external metabolites is marked as both')
    parser.add_argument('--hide', type=str, default='',
                        help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that are transformed into internal metabolites by adding bidirectional exchange reactions')
    parser.add_argument('--prohibit', type=str, default='',
                        help='EXPERIMENTAL. Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that are transformed into internal metabolites without adding bidirectional exchange reactions.'
                             'This metabolite can therefore not be used as input nor output.')
    parser.add_argument('--hide_all_in_or_outputs', type=str, default='',
                        help='Option is only available if --direct is chosen. String that is either empty, input, or output. If it is inputs or outputs, after splitting metabolites, all inputs or outputs are hidden (objective is always excluded)')
    parser.add_argument('--iterative', type=str2bool, default=False,
                        help='Enable iterative conversion mode enumeration (helps on large, dense networks) (default: false)')
    parser.add_argument('--only_rays', type=str2bool, default=False,
                        help='Enable to only return extreme rays, and not elementary modes. This describes the full conversion space, but not all biologically relevant minimal conversions. See (Urbanczik, 2005) (default: false)')
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='Enable to show detailed console output (default: true)')
    parser.add_argument('--scei', type=str2bool, default=True, help='Enable to use SCEI compression (default: true)')
    parser.add_argument('--compare', type=str2bool, default=False,
                        help='Enable to compare output of direct vs indirect')
    parser.add_argument('--job_size', type=int, default=1, help='Number of LPs per multiprocessing job')
    parser.add_argument('--sort_order', type=str, default='min_adj',
                        help='Order in which internal metabolites should be set to zero. Default is to minimize the added adjacencies, other options are: min_lp, max_lp_per_adj, min_connections')
    parser.add_argument('--intermediate_cone_path', type=str, default='',
                        help='Filename where intermediate cone result can be found. If an empty string is given (default), then no intermediate result is picked up and the calculation is done in full')
    parser.add_argument('--manual_override', type=str, default='',
                       help='Index indicating which metabolite should be intersected in first step. Advanced option, can be used in combination with --intermediate_cone_path, to pick a specific intersection at a specific time.')

    args = parser.parse_args()

    with HiddenPrints():
        if args.model_path == '':
            mp_print('No model given, please specify --model_path')
            exit()

        if len(args.inputs) or len(args.outputs):
            # Disable automatic determination of external metabolite direction if lists are given manually
            args.auto_direction = False

        if args.iterative:
            # Only compress when flag is enabled, and when not performing iterative enumeration.
            # Iterative enumeration performs compression after the iteration steps.
            args.compress = False

        model_path = args.model_path

    if args.intermediate_cone_path:
        check_if_intermediate_cone_exists(args.intermediate_cone_path)

    if args.compare or args.direct:
        from mpi4py import MPI
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        with HiddenPrints():  # Store original network, for unhide step
            network = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                                 determine_inputs_outputs=args.auto_direction,
                                                 skip_external_reactions=True,
                                                 external_compartment=args.external_compartment)

            debug_tags = []
            # add_debug_tags(network)

            adj = get_metabolite_adjacency(network.N)

            if not args.auto_direction:
                set_inoutputs(args.inputs, args.outputs, network)

            if args.hide:
                hide_indices = [int(index) for index in args.hide.split(',') if len(index)]
                network.hide(hide_indices)

            if args.prohibit:
                prohibit_indices = [int(index) for index in args.prohibit.split(',') if len(index)]
                network.prohibit(prohibit_indices)

            if args.print_reactions:
                mp_print('Reactions%s:' % (' before compression' if args.compress else ''))
                for index, item in enumerate(network.reactions):
                    mp_print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

            if args.print_metabolites:
                mp_print('Metabolites%s:' % (' before compression' if args.compress else ''))
                for index, item in enumerate(network.metabolites):
                    print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

            orig_ids = [m.id for m in network.metabolites]
            orig_N = network.N

            # for i, r in enumerate(network.reactions):
            #     mp_print("\n%s:" % (r.id))
            #     for j in range(len(network.N[:, i])):
            #         nr = network.N[j, i];
            #         if nr != 0:
            #             mp_print("%s: %d" % (network.metabolites[j].id, nr))

            # Split metabolites in input and output
            network.split_in_out(args.only_rays)

            if args.hide_all_in_or_outputs:
                hide_indices = [ind for ind, metab in enumerate(network.metabolites) if
                                (metab.is_external) & (metab.direction == args.hide_all_in_or_outputs) & (
                                    not metab.id == 'objective_out')]
                network.hide(hide_indices)

            if args.compress:
                network.compress(verbose=args.verbose, SCEI=args.scei)

            removed = 0
            for i in np.flip(range(network.N.shape[0]), 0):
                if sum(abs(network.N[i])) == 0:
                    if not network.metabolites[i].is_external:
                        network.drop_metabolites([i], force_external=True)
                        removed += 1
            mp_print("Removed %d metabolites that were not in any reactions" % removed)

            network.split_reversible()
            network.N = np.transpose(redund(np.transpose(network.N)))

            R, network, external_cycles = remove_cycles(network.N, network)
            R = network.N
            n_reac_according_to_N = network.N.shape[1]
            removable_reacs = np.arange(n_reac_according_to_N, len(network.reactions))
            network.drop_reactions(removable_reacs)
            network = compress_after_cycle_removing(network)
            R = network.N

            external = np.asarray(network.external_metabolite_indices())
            internal = np.setdiff1d(range(R.shape[0]), external)

        T_intersected, ids = intersect_directly(R, internal, network, verbose=args.verbose, lps_per_job=args.job_size,
                                                sort_order=args.sort_order, manual_override=args.manual_override,
                                                intermediate_cone_path=args.intermediate_cone_path)

        if len(external_cycles):
            external_cycles_array = to_fractions(np.zeros((T_intersected.shape[0],len(external_cycles))))
            for ind, cycle in enumerate(external_cycles):
                for cycle_metab in cycle:
                    metab_ind = [ind for ind, metab in enumerate(ids) if metab == cycle_metab][0]
                    external_cycles_array[metab_ind, ind] = cycle[cycle_metab]

            T_intersected = np.concatenate((T_intersected, external_cycles_array, -external_cycles_array), axis=1)

        print_ecms_direct(T_intersected, ids)

        # save to file
        if MPI.COMM_WORLD.Get_rank() == 0:
            try:
                np.savetxt(args.out_path, np.transpose(T_intersected), delimiter=',', header=','.join(ids), comments='')
            except OverflowError:
                norm_T_intersected = normalize_columns(T_intersected)
                np.savetxt(args.out_path, np.transpose(norm_T_intersected), delimiter=',', header=','.join(ids),
                           comments='')

        end = time()
        mp_print('Ran (direct) in %f seconds with %d processes' % (end - start, MPI.COMM_WORLD.Get_size()))

    # input("waiting")
    if args.compare or not args.direct:
        network = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                             determine_inputs_outputs=args.auto_direction,
                                             skip_external_reactions=True,
                                             external_compartment=args.external_compartment)

        debug_tags = []
        # add_debug_tags(network)

        adj = get_metabolite_adjacency(network.N)

        if not args.auto_direction:
            set_inoutputs(args.inputs, args.outputs, network)

        if args.hide:
            hide_indices = [int(index) for index in args.hide.split(',') if len(index)]
            network.hide(hide_indices)

        if args.prohibit:
            prohibit_indices = [int(index) for index in args.prohibit.split(',') if len(index)]
            network.prohibit(prohibit_indices)

        if args.print_reactions:
            mp_print('Reactions%s:' % (' before compression' if args.compress else ''))
            for index, item in enumerate(network.reactions):
                mp_print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

        if args.print_metabolites:
            mp_print('Metabolites%s:' % (' before compression' if args.compress else ''))
            for index, item in enumerate(network.metabolites):
                mp_print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

        orig_ids = [m.id for m in network.metabolites]
        orig_N = network.N

        # for i, r in enumerate(network.reactions):
        #     mp_print("\n%s:" % (r.id))
        #     for j in range(len(network.N[:, i])):
        #         nr = network.N[j, i];
        #         if nr != 0:
        #             mp_print("%s: %d" % (network.metabolites[j].id, nr))

        if args.compress:
            network.compress(verbose=args.verbose, SCEI=args.scei)

        if args.print_reactions and args.compress:
            mp_print('Reactions (after compression):')
            for index, item in enumerate(network.reactions):
                mp_print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

        if args.print_metabolites and args.compress:
            mp_print('Metabolites (after compression):')
            for index, item in enumerate(network.metabolites):
                mp_print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

        if args.iterative:
            cone = network.uncompress(
                iterative_conversion_cone(network, only_rays=args.only_rays, verbose=args.verbose))
        else:
            cone = network.uncompress(get_conversion_cone(network.N, network.external_metabolite_indices(),
                                                          network.reversible_reaction_indices(),
                                                          input_metabolites=network.input_metabolite_indices(),
                                                          output_metabolites=network.output_metabolite_indices(),
                                                          verbose=args.verbose, only_rays=args.only_rays))

        np.savetxt(args.out_path, cone, delimiter=',')

        if args.print_conversions:
            print_ECMs(cone, debug_tags, network, orig_N, args.add_objective_metabolite, args.check_feasibility)

        if args.check_bijection:
            check_bijection(cone, network, model_path, args)

        end = time()
        mp_print('Ran in %f seconds' % (end - start))

    if args.compare:
        metabolites = [m.id for m in network.metabolites]
        for i in range(cone.shape[1]):
            if sum(abs(cone[:, i])) == 0:
                if network.uncompressed_metabolite_ids[i] in metabolites:
                    metabolite_nr = [m.id for m in network.metabolites].index(network.uncompressed_metabolite_ids[i])
                    network.drop_metabolites([metabolite_nr], force_external=True)
        cone_without_zeroes = cone[:, [sum(abs(cone[:, i])) != 0 for i in range(cone.shape[1])]]
        ids = list(np.array(ids)[[sum(abs(T_intersected[i, :])) != 0 for i in range(T_intersected.shape[0])]])
        T_without_zeroes = T_intersected[[sum(abs(T_intersected[i, :])) != 0 for i in range(T_intersected.shape[0])], :]

        # align metabolites
        metabolites = [m.id for m in network.metabolites]
        aligned_R = T_without_zeroes.copy()
        for i in range(len(metabolites)):
            aligned_R[i, :] = T_without_zeroes[ids.index(metabolites[i]), :]

        match, ecms_first_min_ecms_second, ecms_second_min_ecms_first = check_bijection_Erik(aligned_R, np.transpose(
            cone_without_zeroes), network)
        if match:
            mp_print("\n\t\tMatch\n")
        else:
            mp_print("\n\t\tNO match\n")
            mp_print("\nFirst minus second:")
            for i in range(ecms_first_min_ecms_second.shape[1]):
                pass
