import numpy as np
from time import time
from scipy.optimize import linprog
from argparse import ArgumentParser, ArgumentTypeError
from sklearn.preprocessing import normalize

from ecmtool.helpers import get_efms, get_metabolite_adjacency, redund
from ecmtool.intersect_directly import intersect_directly, print_ecms_direct, remove_cycles
from ecmtool.network import extract_sbml_stoichiometry
from ecmtool.conversion_cone import get_conversion_cone, iterative_conversion_cone, unique


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def print_ECMs(cone, debug_tags, network, orig_N, add_objective_metabolite, check_feasibility):
    for index, ecm in enumerate(cone):
        # Normalise by objective metabolite, if applicable
        objective_index = -1 - len(debug_tags)
        objective = ecm[objective_index]
        if add_objective_metabolite and objective > 0:
            ecm /= objective

        metabolite_ids = [met.id for met in network.metabolites] if not network.compressed else network.uncompressed_metabolite_ids

        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%s\t\t->\t%.4f' % (metabolite_ids[metabolite_index], stoichiometry_val))

        if check_feasibility:
            allowed_error = 10**-6
            solution = linprog(c=[0] * orig_N.shape[1], A_eq=orig_N, b_eq=cone[index, :],
                               bounds=[(-1000, 1000)] * orig_N.shape[1], options={'tol': allowed_error})
            print('ECM satisfies stoichiometry' if solution.status == 0 else 'ECM does not satisfy stoichiometry')


def remove_close_vectors(matrix, threshold=10**-6, verbose=True):
    i = 0
    new_matrix = matrix

    if verbose:
        print('Removing vectors with small distance to others')

    while i < new_matrix.shape[0]:
        temp_matrix = new_matrix
        unique_indices = range(i + 1) + [index + i + 1 for index in find_matching_vector_indices(temp_matrix[i, :], temp_matrix[i + 1:, :], near=False, threshold=threshold)]

        if verbose:
            print('%.2f%% (removed %d/%d)' % (100*float(i)/new_matrix.shape[0], matrix.shape[0] - new_matrix.shape[0], matrix.shape[0]))

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


def check_bijection(conversion_cone, network, model_path, args, verbose=True):
    # Add exchange reactions because EFMtool needs them
    full_model = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                            determine_inputs_outputs=args.auto_direction,
                                            skip_external_reactions=True)
    if args.check_bijection:
        set_inoutputs(args.inputs, args.outputs, full_model)

    ex_N = full_model.N
    identity = np.identity(len(full_model.metabolites))
    reversibilities = [reaction.reversible for reaction in full_model.reactions]

    for index, metabolite in enumerate(full_model.metabolites):
        if metabolite.is_external:
            reaction = identity[:, index] if metabolite.direction != 'output' else -identity[index]
            ex_N = np.append(ex_N, np.transpose([reaction]), axis=1)
            reversibilities.append(True if metabolite.direction == 'both' else False)

    efms = get_efms(ex_N, reversibilities)
    if verbose:
        print('Calculating ECMs from EFMs')
    efm_ecms = np.transpose(np.dot(np.asarray(ex_N, dtype='float'), np.transpose(efms)))
    if verbose:
        print('Removing non-unique ECMs')
    efm_ecms_normalised = normalize(efm_ecms, axis=1)
    efm_ecms_normalised = remove_close_vectors(efm_ecms_normalised)
    efm_ecms_unique = unique(efm_ecms_normalised)
    efm_ecms_unique = redund(efm_ecms_unique)

    ecmtool_ecms_normalised = normalize(conversion_cone, axis=1)

    if verbose:
        print('Found %d efmtool-calculated ECMs, and %d ecmtool ones' % (len(efm_ecms_unique), len(ecmtool_ecms_normalised)))
        print('Checking bijection')

    is_bijection = True

    for index, ecm in enumerate(efm_ecms_unique):
        if verbose:
            print('Checking %d/%d (round 1/2)' % (index+1, len(efm_ecms_unique)))
        close_vectors = find_matching_vector_indices(ecm, ecmtool_ecms_normalised, near=True)
        if len(close_vectors) != 1:
            is_bijection = False
            print('\nCalculated ECM #%d not uniquely in enumerated list (got %d matches):' % (index, len(close_vectors)))
            for metabolite_index, stoichiometry_val in enumerate(ecm):
                if stoichiometry_val != 0.0:
                    print('%d %s\t\t->\t%.4f' % (
                    metabolite_index, network.uncompressed_metabolite_names[metabolite_index], stoichiometry_val))

    for index, ecm in enumerate(ecmtool_ecms_normalised):
        if verbose:
            print('Checking %d/%d (round 2/2)' % (index+1, len(ecmtool_ecms_normalised)))
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
        print(
            'No output metabolites given or determined from model. All non-input metabolites will be defined as outputs.')
        outputs = np.setdiff1d(network.external_metabolite_indices(), inputs)

    network.set_inputs(inputs)
    network.set_outputs(outputs)
    return

if __name__ == '__main__':
    start = time()

    parser = ArgumentParser(description='Calculate Elementary Conversion Modes from an SBML model. For medium-to large networks, be sure to define --inputs and --outputs. This reduces the enumeration problem complexity considerably.')
    parser.add_argument('--model_path', type=str, default='models/e_coli_core.xml', help='Relative or absolute path to an SBML model .xml file')
    parser.add_argument('--direct', type=str2bool, default=True, help='Enable to intersect with equalities directly')
    parser.add_argument('--compress', type=str2bool, default=True, help='Perform compression to which the conversions are invariant, and reduce the network size considerably (default: True)')
    parser.add_argument('--out_path', default='conversion_cone.csv', help='Relative or absolute path to the .csv file you want to save the calculated conversions to (default: conversion_cone.csv)')
    parser.add_argument('--add_objective_metabolite', type=str2bool, default=True, help='Add a virtual metabolite containing the stoichiometry of the objective function of the model (default: true)')
    parser.add_argument('--check_feasibility', type=str2bool, default=False, help='For each found ECM, verify that a feasible flux exists that produces it (default: false)')
    parser.add_argument('--check_bijection', type=str2bool, default=False, help='Verify completeness of found ECMs by calculating ECMs from EFMs and proving bijection (don\'t use on large networks) (default: false)')
    parser.add_argument('--print_metabolites', type=str2bool, default=True, help='Print the names and IDs of metabolites in the (compressed) metabolic network (default: true)')
    parser.add_argument('--print_reactions', type=str2bool, default=False, help='Print the names and IDs of reactions in the (compressed) metabolic network (default: true)')
    parser.add_argument('--auto_direction', type=str2bool, default=True, help='Automatically determine external metabolites that can only be consumed or produced (default: true)')
    parser.add_argument('--inputs', type=str, default='', help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that can only be consumed')
    parser.add_argument('--outputs', type=str, default='', help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that can only be produced')
    parser.add_argument('--hide', type=str, default='', help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that are transformed into internal metabolites by adding bidirectional exchange reactions')
    parser.add_argument('--iterative', type=str2bool, default=False, help='Enable iterative conversion mode enumeration (helps on large, dense networks) (default: false)')
    parser.add_argument('--only_rays', type=str2bool, default=False, help='Enable to only return extreme rays, and not elementary modes. This describes the full conversion space, but not all biologically relevant minimal conversions. See (Urbanczik, 2005) (default: false)')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Enable to show detailed console output (default: true)')
    parser.add_argument('--scei', type=str2bool, default=True, help='Enable to use SCEI compression (default: true)')
    args = parser.parse_args()

    if args.model_path == '':
        print('No model given, please specify --model_path')
        exit()

    if len(args.inputs) or len(args.outputs):
        # Disable automatic determination of external metabolite direction if lists are given manually
        args.auto_direction = False

    if args.iterative:
        # Only compress when flag is enabled, and when not performing iterative enumeration.
        # Iterative enumeration performs compression after the iteration steps.
        args.compress = False

    model_path = args.model_path

    network = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                         determine_inputs_outputs=args.auto_direction,
                                         skip_external_reactions=True)

    debug_tags = []
    # add_debug_tags(network)

    adj = get_metabolite_adjacency(network.N)

    if not args.auto_direction:
        set_inoutputs(args.inputs, args.outputs, network)

    if args.hide:
        hide_indices = [int(index) for index in args.hide.split(',') if len(index)]
        network.hide(hide_indices)

    if args.print_reactions:
        print('Reactions%s:' % (' before compression' if args.compress else ''))
        for index, item in enumerate(network.reactions):
            print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

    if args.print_metabolites:
        print('Metabolites%s:' % (' before compression' if args.compress else ''))
        for index, item in enumerate(network.metabolites):
            print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

    orig_ids = [m.id for m in network.metabolites]
    orig_N = network.N

    if args.direct and not args.only_rays:
        network.split_in_out()

    for i, r in enumerate(network.reactions):
        print("\n%s:" % (r.id))
        for j in range(len(network.N[:, i])):
            nr = network.N[j, i];
            if nr != 0:
                print("%s: %d" % (network.metabolites[j].id, nr))

    if args.compress:
        network.compress(verbose=args.verbose, SCEI=args.scei)

    network.N = np.transpose(redund(np.transpose(network.N)))
    for i in np.flip(range(network.N.shape[0]), 0):
        if sum(abs(network.N[i])) == 0:
            network.drop_metabolites([i], force_external=True)

    if args.direct:
        # DEBUG try floats everywhere
        #network.N = np.array(network.N, dtype='float')

        network.split_reversible()
        #if not args.only_rays:
        #    network.split_in_out()
        R, deleted = remove_cycles(network.N, network)

        external = np.asarray(network.external_metabolite_indices())
        internal = np.setdiff1d(range(R.shape[0]), external)
        T_intersected = intersect_directly(R, internal, network, verbose=args.verbose)

        print_ecms_direct(T_intersected, network.external_metabolite_indices(), network.metabolites)
        end = time()
        print('Ran (direct) in %f seconds' % (end - start))
    else:
        if args.print_reactions and args.compress:
            print('Reactions (after compression):')
            for index, item in enumerate(network.reactions):
                print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

        if args.print_metabolites and args.compress:
            print('Metabolites (after compression):')
            for index, item in enumerate(network.metabolites):
                print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

        if args.iterative:
            cone = network.uncompress(iterative_conversion_cone(network, only_rays=args.only_rays, verbose=args.verbose))
        else:
            pre_cone = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                                       input_metabolites=network.input_metabolite_indices(),
                                       output_metabolites=network.output_metabolite_indices(), verbose=args.verbose, only_rays=args.only_rays)
            cone = network.uncompress(pre_cone)

        np.savetxt(args.out_path, cone, delimiter=',')

        print_ECMs(cone, debug_tags, network, orig_N, args.add_objective_metabolite, args.check_feasibility)

        if args.check_bijection:
            check_bijection(cone, network, model_path, args)

        end = time()
        print('Ran in %f seconds' % (end - start))
        pass
