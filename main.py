import os
import sys
from argparse import ArgumentParser, ArgumentTypeError
from time import time

import numpy as np
from scipy.optimize import linprog

from subprocess import run

from ecmtool import mpi_wrapper
from ecmtool.conversion_cone import get_conversion_cone, iterative_conversion_cone
from ecmtool.helpers import get_metabolite_adjacency, redund, to_fractions
from ecmtool.helpers import mp_print, unsplit_metabolites, print_ecms_direct, normalize_columns
from ecmtool.network import extract_sbml_stoichiometry, add_reaction_tags


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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

        mp_print('\nECM #%d:' % (index + 1))
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                mp_print('%s\t\t->\t%.4f' % (metabolite_ids[metabolite_index], stoichiometry_val))

        if check_feasibility:
            satisfied = ecm_satisfies_stoichiometry(orig_N, cone[index, :])
            mp_print('ECM satisfies stoichiometry' if satisfied else 'ECM does not satisfy stoichiometry')


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
                'Metabolite %s was marked as both only input and only output. It is set to both.' % (
                    network.metabolites[ind].id))
        network.set_both(np.intersect1d(inputs, outputs))
    return


if __name__ == '__main__':
    start = time()

    parser = ArgumentParser(
        description='Calculate Elementary Conversion Modes from an SBML model. For medium-to large networks, be sure to define --inputs and --outputs. This reduces the enumeration problem complexity considerably.')
    parser.add_argument('--model_path', type=str, default='models/active_subnetwork_KO_5.xml',
                        help='Relative or absolute path to an SBML model .xml file')
    parser.add_argument('--direct', type=str2bool, default=False, help='Enable to intersect with equalities directly. Direct intersection works better than indirect when many metabolites are hidden, and on large networks (default: False)')
    parser.add_argument('--compress', type=str2bool, default=True,
                        help='Perform compression to which the conversions are invariant, and reduce the network size considerably (default: True)')
    parser.add_argument('--remove_infeasible', type=str2bool, default=True,
                        help='Remove reactions that cannot carry flux dsquring compression. Switch off when this gives rise to numerical linear algebra problems. (default: True)')
    parser.add_argument('--out_path', default='conversion_cone.csv',
                        help='Relative or absolute path to the .csv file you want to save the calculated conversions to (default: conversion_cone.csv)')
    parser.add_argument('--add_objective_metabolite', type=str2bool, default=True,
                        help='Add a virtual metabolite containing the stoichiometry of the objective function of the model (default: true)')
    parser.add_argument('--print_metabolites', type=str2bool, default=True,
                        help='Print the names and IDs of metabolites in the (compressed) metabolic network (default: true)')
    parser.add_argument('--print_reactions', type=str2bool, default=False,
                        help='Print the names and IDs of reactions in the (compressed) metabolic network (default: true)')
    parser.add_argument('--print_conversions', type=str2bool, default=False,
                        help='Print the calculated conversion modes (default: false)')
    parser.add_argument('--use_external_compartment', type=str, default=None,
                        help='If a string is given, this string indicates how the external compartment in metabolite_ids of SBML-file is marked. By default, dead-end reaction-detection is used to find external metabolites, and no compartment-information. Please check if external compartment detection works by checking metabolite information before compression and with --primt metabolites true')
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
                        help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that are transformed into internal metabolites without adding bidirectional exchange reactions.'
                             'This metabolite can therefore not be used as input nor output.')
    parser.add_argument('--tag', type=str, default='',
                        help='Comma-separated list of reaction indices, as given by --print_reactions true (before compression), that will be tagged with new virtual metabolites, such that the reaction flux appears in ECMs.')
    parser.add_argument('--hide_all_in_or_outputs', type=str, default='',
                        help='String that is either empty, input, or output. If it is inputs or outputs, after splitting metabolites, all inputs or outputs are hidden (objective is always excluded)')
    parser.add_argument('--iterative', type=str2bool, default=False,
                        help='Enable iterative conversion mode enumeration (helps on large, dense networks) (default: false)')
    parser.add_argument('--only_rays', type=str2bool, default=False,
                        help='Enable to only return extreme rays, and not elementary modes. This describes the full conversion space, but not all biologically relevant minimal conversions. See (Urbanczik, 2005) (default: false)')
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='Enable to show detailed console output (default: true)')
    parser.add_argument('--splitting_before_polco', type=str2bool, default=True,
                        help='Enables splitting external metabolites by making virtual input and output metabolites before starting the computation. Setting to false would do the splitting after first computation step. Which method is faster is complicated and model-dependent. (default: true)')
    parser.add_argument('--redund_after_polco', type=str2bool, default=True,
                        help='(Indirect intersection only) Enables redundant row removal from inequality description of dual cone. Works well with models with relatively many internal metabolites, and when running parrallelized computation using MPI (default: true)')
    parser.add_argument('--scei', type=str2bool, default=True, help='Enable to use SCEI compression (default: true)')
    parser.add_argument('--sort_order', type=str, default='min_adj',
                        help='Order in which internal metabolites should be set to zero. Default is to minimize the added adjacencies, other options are: min_lp, max_lp_per_adj, min_connections')
    parser.add_argument('--intermediate_cone_path', type=str, default='',
                        help='Filename where intermediate cone result can be found. If an empty string is given (default), then no intermediate result is picked up and the calculation is done in full')
    parser.add_argument('--manual_override', type=str, default='',
                        help='Index indicating which metabolite should be intersected in first step. Advanced option, can be used in combination with --intermediate_cone_path, to pick a specific intersection at a specific time.')
    
    parser.add_argument('--polco', type=str2bool, default=False,
                        help='Uses polco instead of mplrs for extreme ray enumeration (default: false)')
    parser.add_argument('--processes', type=int, default=3,
                        help='Numer of processes for calculations (default: 3 - minimum required for mplrs)')
    parser.add_argument('--jvm_mem', type=int, default=None, nargs='*', action='store',
                        help='Two values given the minimum and maximum memeory for java machine in GB e.g. 50 300 (default: maximum memory available)')
    parser.add_argument('--path2mplrs', type=str, default=None,
                        help='if mplrs binary is not accessable via PATH variable "mplrs", the absolute path to the binary can be provided with "--path2mplrs" e.g. "--path2mplrs /home/user/mplrs/lrslib-071b/mplrs" ')

    args = parser.parse_args()
    
    if args.jvm_mem is not None and len(args.jvm_mem) not in (0, 2):
        parser.error('Either give no values for jvm_mem, or two - "minGB maxGB" e.g. 50 200, not {}.'.format(len(args.jvm_mem)))
   
    if args.polco is False and args.path2mplrs is None:
        try:
            mplrs_check = run(['mplrs'],capture_output=True)
            if args.verbose is True:
                print('Found mplrs path variable')
        except:
            print('\x1b[0;31;40m' + 'WARNING1: mplrs NOT found' + '\x1b[0m')
            print('\x1b[0;31;40m' + 'Make sure mplrs is installed properly, see http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html' + '\x1b[0m')
            print('\x1b[0;31;40m' + 'Make sure mplrs is added to the PATH variable or provide absolute path to mplrs binary via command line argument --path2mplrs' + '\x1b[0m')
            exit(-1)

    if args.polco is False and args.path2mplrs is not None:
        path2mplrs = args.path2mplrs
        try:
            mplrs_check = run([path2mplrs],capture_output=True)
            if args.verbose is True:
                print('Found mplrs')
        except:
            print('\x1b[0;31;40m' + 'WARNING2: mplrs NOT found' + '\x1b[0m')
            print('\x1b[0;31;40m' + 'Make sure mplrs is installed properly, see http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html' + '\x1b[0m')
            print('\x1b[0;31;40m' + 'Make sure mplrs is added to the PATH variable or provide absolute path to mplrs binary via command line argument --path2mplrs' + '\x1b[0m')
            exit(-1)
            
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

    if sys.platform.startswith('win32'):
        mp_print('!!\n'
                 'Running ecmtool on Linux might be faster than on Windows, because of external dependencies.\n'
                 'To reap the benefits of parallelization and the use of cython, please use a virtual Linux machine'
                 'or a Linux computing cluster.\n')

    model_path = args.model_path

    network = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                         determine_inputs_outputs=args.auto_direction,
                                         skip_external_reactions=True,
                                         use_external_compartment=args.use_external_compartment)

    tagged_reaction_indices = []

    adj = get_metabolite_adjacency(network.N)

    if not args.auto_direction:
        set_inoutputs(args.inputs, args.outputs, network)

    if args.hide:
        hide_indices = [int(index) for index in args.hide.split(',') if len(index)]
        network.hide(hide_indices)

    if args.prohibit:
        prohibit_indices = [int(index) for index in args.prohibit.split(',') if len(index)]
        network.prohibit(prohibit_indices)

    tag_ids = []
    if args.tag:
        tagged_reaction_indices = [int(index) for index in args.tag.split(',') if len(index)]
        tag_ids = add_reaction_tags(network, tagged_reaction_indices)

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

    if args.direct:
        from ecmtool.intersect_directly_mpi import intersect_directly, remove_cycles, \
            compress_after_cycle_removing, check_if_intermediate_cone_exists

        # Check if intermediate cone exists at the given location
        if args.intermediate_cone_path:
            check_if_intermediate_cone_exists(args.intermediate_cone_path)

    if args.direct or args.splitting_before_polco:
        # Split metabolites in input and output
        network.split_in_out(args.only_rays)
        cycle_removal_boolean = True if not args.only_rays else False
    else:
        cycle_removal_boolean = False

    if args.hide_all_in_or_outputs:
        if not args.direct and not args.splitting_before_polco:
            network.split_in_out(args.only_rays)

        hide_indices = [ind for ind, metab in enumerate(network.metabolites) if
                        (metab.is_external) & (metab.direction == args.hide_all_in_or_outputs) & (
                            not metab.id == 'objective_virtout') & (
                                    metab.id.replace("_virtin", "").replace("_virtout", "") not in tag_ids)]
        network.hide(hide_indices)

    if args.compress:
        network.compress(verbose=args.verbose, SCEI=args.scei, cycle_removal=cycle_removal_boolean,
                         remove_infeasible=args.remove_infeasible)

    if args.direct:
        network.split_reversible()
        network.N = np.transpose(redund(np.transpose(network.N)))

        R, network, external_cycles = remove_cycles(network.N, network)
        n_reac_according_to_N = network.N.shape[1]
        removable_reacs = np.arange(n_reac_according_to_N, len(network.reactions))
        network.drop_reactions(removable_reacs)
        network = compress_after_cycle_removing(network)

    if args.print_reactions and args.compress:
        mp_print('Reactions (after compression):')
        for index, item in enumerate(network.reactions):
            mp_print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

    if args.print_metabolites and args.compress:
        mp_print('Metabolites (after compression):')
        for index, item in enumerate(network.metabolites):
            mp_print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

    if args.direct:
        # Direct intersection method
        R = network.N
        external = np.asarray(network.external_metabolite_indices())
        internal = np.setdiff1d(range(R.shape[0]), external)
        T_intersected, ids = intersect_directly(R, internal, network, verbose=args.verbose,
                                                sort_order=args.sort_order, manual_override=args.manual_override,
                                                intermediate_cone_path=args.intermediate_cone_path)
        if external_cycles:
            external_cycles_array = to_fractions(np.zeros((T_intersected.shape[0], len(external_cycles))))
            for ind, cycle in enumerate(external_cycles):
                for cycle_metab in cycle:
                    metab_ind = [ind for ind, metab in enumerate(ids) if metab == cycle_metab][0]
                    external_cycles_array[metab_ind, ind] = cycle[cycle_metab]

            T_intersected = np.concatenate((T_intersected, external_cycles_array, -external_cycles_array), axis=1)

        cone = np.transpose(T_intersected)
    else:
        if args.iterative:
            # Indirect iterative enumeration
            cone = network.uncompress(
                iterative_conversion_cone(network, only_rays=args.only_rays, verbose=args.verbose))
        else:
            # Indirect enumeration
            cone = get_conversion_cone(network.N, network.external_metabolite_indices(),
                                       network.reversible_reaction_indices(),
                                       input_metabolites=network.input_metabolite_indices(),
                                       output_metabolites=network.output_metabolite_indices(),
                                       verbose=args.verbose, only_rays=args.only_rays,
                                       redund_after_polco=args.redund_after_polco,
                                       polco=args.polco, processes=args.processes, jvm_mem=args.jvm_mem,
                                       path2mplrs=args.path2mplrs)

            # if external_cycles:
            #     T_intersected = np.transpose(cone)
            #     external_cycles_array = to_fractions(np.zeros((T_intersected.shape[0], len(external_cycles))))
            #     for ind, cycle in enumerate(external_cycles):
            #         for cycle_metab in cycle:
            #             metab_ind = [ind for ind, metab in enumerate(ids) if metab == cycle_metab][0]
            #             external_cycles_array[metab_ind, ind] = cycle[cycle_metab]
            #
            #     T_intersected = np.concatenate((T_intersected, external_cycles_array, -external_cycles_array), axis=1)
            #     cone = np.transpose(T_intersected)

        cone_transpose, ids = unsplit_metabolites(np.transpose(cone), network)
        cone = np.transpose(cone_transpose)
        #
        internal_ids = []
        for metab in network.metabolites:
            if not metab.is_external:
                id_ind = [ind for ind, id in enumerate(ids) if id == metab.id]
                if len(id_ind):
                    internal_ids.append(id_ind[0])

        ids = list(np.delete(ids, internal_ids))
        cone = np.delete(cone, internal_ids, axis=1)

    if mpi_wrapper.is_first_process():
        try:
            np.savetxt(args.out_path, cone, delimiter=',', header=','.join(ids), comments='')
        except OverflowError:
            normalised = np.transpose(normalize_columns(np.transpose(cone)))
            np.savetxt(args.out_path, normalised, delimiter=',', header=','.join(ids), comments='')

            
    if args.verbose is True:
        print('Found %s ECMs' % cone.shape[0])
          
    if args.print_conversions is True:
        print_ecms_direct(np.transpose(cone), ids)
    
    end = time()
    mp_print('Ran in %f seconds' % (end - start))
    os._exit(0)
