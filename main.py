import os
import pickle
import sys
from argparse import ArgumentParser, ArgumentTypeError
from time import time

import numpy as np
from scipy.optimize import linprog

from subprocess import run

from ecmtool import mpi_wrapper
from ecmtool.conversion_cone import calculate_linearities, calc_C0_dual_extreme_rays, calc_H, \
    calc_C_extreme_rays, post_process_rays
from ecmtool.helpers import get_metabolite_adjacency, redund, to_fractions, prep_mplrs_input, execute_mplrs, \
    process_mplrs_ouput
from ecmtool.helpers import mp_print, unsplit_metabolites, print_ecms_direct, normalize_columns, uniqueReadWrite
from ecmtool.intersect_directly_mpi import intersect_directly
from ecmtool.network import extract_sbml_stoichiometry, add_reaction_tags, Network


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
            print(
                'Support: ' + str([index for index, _ in enumerate(solution['x']) if abs(solution['x'][index]) > 1e-6]))

    return in_cone


def set_inoutputs(inputs, outputs, network):
    inputs = [int(index) for index in inputs.split(',') if len(index)]
    outputs = [int(index) for index in outputs.split(',') if len(index)]
    if len(outputs) < 1 and len(inputs) >= 1:
        # If no outputs are given, define all external metabolites that are not inputs as outputs
        mp_print(
            'No output metabolites given or determined from model. All non-input metabolites will be defined as outputs.')
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


def preprocess_sbml(args):
    model_path = args.model_path

    network = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                         determine_inputs_outputs=args.auto_direction,
                                         skip_external_reactions=True,
                                         use_external_compartment=args.use_external_compartment)

    tagged_reaction_indices = []
    external_cycles = None

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
        # Initialise mpi4py only here, because it can not be started when using mplrs due to
        # only being able to run one instance at a time, and mpi4py creates an instance on import.
        mpi_wrapper.mpi_init(mplrs_present=mplrs_present)

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

    return network, external_cycles


def restore_data(filename):
    with open(os.path.join('ecmtool', 'tmp', filename), 'rb') as file:
        data = pickle.load(file)
    return data


def save_data(data, filename):
    with open(os.path.join('ecmtool', 'tmp', filename), 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    start = time()

    parser = ArgumentParser(
        description='Calculate Elementary Conversion Modes from an SBML model. For medium-to large networks, be sure to define --inputs and --outputs. This reduces the enumeration problem complexity considerably.')
    parser.add_argument('command', nargs='?', default='all',
                        help='Optional: run only a single step of ecmtool, continuing from the state of the previous step. \n'
                             'Allowed values (in order of execution): preprocess, direct_intersect (only when --direct true),\n'
                             'calc_linearities, prep_C0_rays, calc_C0_rays, process_C0_rays, calc_H, prep_C_rays, calc_C_rays,\n'
                             'process_C_rays, postprocess, save_ecms. Omit to run all steps.')
    parser.add_argument('--model_path', type=str, default='models/active_subnetwork_KO_5.xml',
                        help='Relative or absolute path to an SBML model .xml file')
    parser.add_argument('--direct', type=str2bool, default=False,
                        help='Enable to intersect with equalities directly. Direct intersection works better than indirect when many metabolites are hidden, and on large networks (default: False)')
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
    parser.add_argument('--make_unique', type=str2bool, default=True,
                        help='Make sure set of ECMs is unique at the end  (default: True)')
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

    if args.polco and args.jvm_mem is not None and len(args.jvm_mem) not in (0, 2):
        parser.error(
            'Either give no values for jvm_mem, or two - "minGB maxGB" e.g. 50 200, not {}.'.format(len(args.jvm_mem)))

    mplrs_present = False
    if (args.polco is False) and (args.command in ['calc_C0_rays', 'calc_C_rays', 'all']):
        path2mplrs = args.path2mplrs if args.path2mplrs is not None else 'mplrs'
        try:
            mplrs_check = run([path2mplrs], capture_output=True)
            mplrs_present = True
            if args.verbose is True:
                mp_print('Found mplrs')
        except:
            mp_print('\x1b[0;31;40m' + 'WARNING: mplrs NOT found' + '\x1b[0m')
            mp_print(
                '\x1b[0;31;40m' + 'Make sure mplrs is installed properly, see http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html' + '\x1b[0m')
            mp_print(
                '\x1b[0;31;40m' + 'Make sure mplrs is added to the PATH variable or provide absolute path to mplrs binary via command line argument --path2mplrs' + '\x1b[0m')
            mp_print('\x1b[0;31;40m' + 'Switching to POLCO' + '\x1b[0m')
            args.polco = True

    if args.model_path == '':
        mp_print('No model given, please specify --model_path')
        exit()

    if len(args.inputs) or len(args.outputs):
        # Disable automatic determination of external metabolite direction if lists are given manually
        args.auto_direction = False

    if sys.platform.startswith('win32'):
        mp_print('!!\n'
                 'Running ecmtool on Linux might be faster than on Windows, because of external dependencies.\n'
                 'To reap the benefits of parallelization and the use of cython, please use a virtual Linux machine'
                 'or a Linux computing cluster.\n')

    if args.command in ['preprocess', 'all']:
        mp_print("\nPreprocessing model.")
        network, external_cycles = preprocess_sbml(args)
        save_data(network, 'network.dat')
        save_data(external_cycles, 'external_cycles.dat')

    if args.direct:
        if args.command in ['direct_intersect', 'all']:
            if 'network' not in locals():
                network = restore_data('network.dat')
            if 'external_cycles' not in locals():
                external_cycles = restore_data('external_cycles.dat')

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
            save_data(cone, 'cone.dat')
    else:
        # Indirect enumeration
        if args.command in ['calc_linearities', 'all']:
            mp_print("\nCalculating linearities in original network.")
            if 'network' not in locals():
                network = restore_data('network.dat')

            linearity_data = calculate_linearities(network.N, network.reversible_reaction_indices(),
                                                   network.external_metabolite_indices(),
                                                   network.input_metabolite_indices(),
                                                   network.output_metabolite_indices(), args.verbose)
            save_data(linearity_data, 'linearity_data.dat')

        if args.polco:
            if args.command in ['calc_C0_rays', 'all']:
                if 'linearity_data' not in locals():
                    linearity_data = restore_data('linearity_data.dat')
                linearities, linearities_deflated, G_rev, G_irrev, amount_metabolites, \
                extended_external_metabolites, in_out_indices = linearity_data

                C0_dual_rays = calc_C0_dual_extreme_rays(linearities, G_rev, G_irrev,
                                                         polco=args.polco, processes=args.processes,
                                                         jvm_mem=args.jvm_mem,
                                                         path2mplrs=args.path2mplrs)
                save_data(C0_dual_rays, 'C0_dual_rays.dat')
        else:
            # Using mplrs for enumeration
            if args.command in ['prep_C0_rays', 'all']:
                mp_print("\nPreparing for first vertex enumeration step.")
                if 'linearity_data' not in locals():
                    linearity_data = restore_data('linearity_data.dat')
                linearities, linearities_deflated, G_rev, G_irrev, amount_metabolites, \
                extended_external_metabolites, in_out_indices = linearity_data

                width_matrix = prep_mplrs_input(np.append(linearities, G_rev, axis=0), G_irrev)
                save_data(width_matrix, 'width_matrix.dat')
            if args.command in ['calc_C0_rays', 'all']:
                mp_print("\nFirst vertex enumeration step.")
                # This step gets skipped when running on a computing cluster,
                # in order to run mplrs directly with mpirun.
                execute_mplrs(processes=args.processes, path2mplrs=args.path2mplrs, verbose=args.verbose)
            if args.command in ['process_C0_rays', 'all']:
                mp_print("\nProcessing results from first vertex enumeration step.")
                if 'width_matrix' not in locals():
                    width_matrix = restore_data('width_matrix.dat')
                C0_dual_rays = process_mplrs_ouput(width_matrix, verbose=args.verbose)
                save_data(C0_dual_rays, 'C0_dual_rays.dat')

        if args.command in ['calc_H', 'all']:
            # Initialise mpi4py only here, because it can not be started when using mplrs due to
            # only being able to run one instance at a time, and mpi4py creates an instance on import.
            mpi_wrapper.mpi_init(mplrs_present=mplrs_present)
            mp_print("\nCalculating H. Adding steady-state, irreversibility constraints, "
                     "then discarding redundant inequalities.")
            if mpi_wrapper.is_first_process():
                if 'C0_dual_rays' not in locals():
                    C0_dual_rays = restore_data('C0_dual_rays.dat')

                if 'linearity_data' not in locals():
                    linearity_data = restore_data('linearity_data.dat')
                linearities, linearities_deflated, G_rev, G_irrev, amount_metabolites, \
                extended_external_metabolites, in_out_indices = linearity_data

                if 'network' not in locals():
                    network = restore_data('network.dat')
                H = calc_H(rays=C0_dual_rays, linearities_deflated=linearities_deflated,
                           external_metabolites=network.external_metabolite_indices(),
                           input_metabolites=network.input_metabolite_indices(),
                           output_metabolites=network.output_metabolite_indices(), in_out_indices=in_out_indices,
                           redund_after_polco=args.redund_after_polco, only_rays=args.only_rays, verbose=args.verbose)
                save_data(H, 'H.dat')
            else:
                H = calc_H()
                exit()

        if args.polco:
            if args.command in ['calc_C_rays', 'all']:
                if 'H' not in locals():
                    H = restore_data('H.dat')
                H_eq, H_ineq, linearity_rays = H
                C_rays = calc_C_extreme_rays(H_eq, H_ineq,
                                             polco=args.polco, processes=args.processes, jvm_mem=args.jvm_mem,
                                             path2mplrs=args.path2mplrs)

                save_data(C_rays, 'C_rays.dat')
        else:
            # Using mplrs for enumeration
            if args.command in ['prep_C_rays', 'all']:
                mp_print("\nPreparing for second vertex enumeration step.")
                if 'H' not in locals():
                    H = restore_data('H.dat')
                H_eq, H_ineq, linearity_rays = H

                width_matrix = prep_mplrs_input(H_eq, H_ineq)
                save_data(width_matrix, 'width_matrix.dat')
            if args.command in ['calc_C_rays', 'all']:
                mp_print("\nPerforming second vertex enumeration step.")
                # This step gets skipped when running on a computing cluster,
                # in order to run mplrs directly with mpirun.
                execute_mplrs(processes=args.processes, path2mplrs=args.path2mplrs, verbose=args.verbose)
            if args.command in ['process_C_rays', 'all']:
                mp_print("\nProcessing results from second vertex enumeration step.")
                if 'width_matrix' not in locals():
                    width_matrix = restore_data('width_matrix.dat')
                C_rays = process_mplrs_ouput(width_matrix, verbose=args.verbose)
                save_data(C_rays, 'C_rays.dat')

        if args.command in ['postprocess', 'all']:
            if 'C_rays' not in locals():
                C_rays = restore_data('C_rays.dat')

            if 'H' not in locals():
                H = restore_data('H.dat')
            H_eq, H_ineq, linearity_rays = H

            if 'linearity_data' not in locals():
                linearity_data = restore_data('linearity_data.dat')
            linearities, linearities_deflated, G_rev, G_irrev, amount_metabolites, \
            extended_external_metabolites, in_out_indices = linearity_data

            if 'network' not in locals():
                network = restore_data('network.dat')

            G = np.transpose(network.N)
            cone = post_process_rays(G, C_rays, linearity_rays, network.external_metabolite_indices(),
                                     extended_external_metabolites,
                                     in_out_indices, amount_metabolites, only_rays=args.only_rays, verbose=args.verbose)
            save_data(cone, 'cone.dat')

    if args.command in ['save_ecms', 'all'] and mpi_wrapper.is_first_process():
        if 'cone' not in locals():
            cone = restore_data('cone.dat')

        if 'network' not in locals():
            network = restore_data('network.dat')

        cone_transpose, ids = unsplit_metabolites(np.transpose(cone), network)
        cone = np.transpose(cone_transpose)

        internal_ids = []
        for metab in network.metabolites:
            if not metab.is_external:
                id_ind = [ind for ind, id in enumerate(ids) if id == metab.id]
                if len(id_ind):
                    internal_ids.append(id_ind[0])

        ids = list(np.delete(ids, internal_ids))
        cone = np.delete(cone, internal_ids, axis=1)

        try:
            np.savetxt(args.out_path, cone, delimiter=',', header=','.join(ids), comments='')
        except OverflowError:
            normalised = np.transpose(normalize_columns(np.transpose(cone)))
            np.savetxt(args.out_path, normalised, delimiter=',', header=','.join(ids), comments='')

        if args.verbose is True:
            mp_print('Found %s ECMs' % cone.shape[0])

        if args.print_conversions is True:
            print_ecms_direct(np.transpose(cone), ids)

        if args.make_unique is True:
            uniqueReadWrite(args.out_path)

    end = time()
    mp_print('Ran in %f seconds' % (end - start))
    # os._exit(0)
