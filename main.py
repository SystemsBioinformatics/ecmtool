from scipy.optimize import linprog

from helpers import *
from time import time
from conversion_cone import get_conversion_cone
from argparse import ArgumentParser, ArgumentTypeError

from network import extract_sbml_stoichiometry


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


# Note [Tom]: Glucose, ammonium, O2, phosphate

def print_ECMs(cone, debug_tags, network, orig_N, add_objective_metabolite, check_feasibility):
    for index, ecm in enumerate(cone):
        # Normalise by objective metabolite, if applicable
        objective_index = -1 - len(debug_tags)
        objective = ecm[objective_index]
        if add_objective_metabolite and objective > 0:
            ecm /= objective
            expanded_c[index, :] /= objective

        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%d %s\t\t->\t%.4f' % (metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

        if check_feasibility:
            allowed_error = 10**-6
            solution = linprog(c=[0] * orig_N.shape[1], A_eq=orig_N, b_eq=expanded_c[index, :],
                               bounds=[(-1000, 1000)] * orig_N.shape[1], options={'tol': allowed_error})
            print('ECM satisfies stoichiometry' if solution.status == 0 else 'ECM does not satisfy stoichiometry')


def check_bijection(conversion_cone, network, model_path, add_objective_metabolite):
    # Add exchange reactions because EFMtool needs them
    full_model = extract_sbml_stoichiometry(model_path, add_objective=add_objective_metabolite,
                                            skip_external_reactions=True)
    ex_N = full_model.N
    identity = np.identity(len(full_model.metabolites))
    reversibilities = [reaction.reversible for reaction in full_model.reactions]

    for index, metabolite in enumerate(full_model.metabolites):
        if metabolite.is_external:
            reaction = identity[:, index] if metabolite.direction != 'output' else -identity[index]
            ex_N = np.append(ex_N, np.transpose([reaction]), axis=1)
            reversibilities.append(True if metabolite.direction == 'both' else False)

    efms = get_efms(ex_N, reversibilities)
    ecms = np.transpose(np.dot(full_model.N, np.transpose(efms[:, :len(full_model.reactions)])))
    ecms_unique = np.unique(np.asarray(ecms, dtype='float64'), axis=0)

    is_bijection = True

    for index, ecm in enumerate(conversion_cone):
        if ecm not in ecms_unique:
            is_bijection = False
            print('\nEnumerated ECM #%d not in calculated list:' % index)
            for metabolite_index, stoichiometry_val in enumerate(ecm):
                if stoichiometry_val != 0.0:
                    print('%d %s\t\t->\t%.4f' % (
                    metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

    for index, ecm in enumerate(ecms_unique):
        if ecm not in conversion_cone:
            is_bijection = False
            print('\nCalculated ECM #%d not in enumerated list:' % index)
            for metabolite_index, stoichiometry_val in enumerate(ecm):
                if stoichiometry_val != 0.0:
                    print('%d %s\t\t->\t%.4f' % (
                    metabolite_index, network.metabolites[metabolite_index].name, stoichiometry_val))

    print('Enumerated ECMs and calculated ECMs are%s bijective' % ('' if is_bijection else ' not'))


if __name__ == '__main__':
    start = time()

    parser = ArgumentParser(description='Calculate Elementary Conversion Modes from an SBML model. For medium-to large networks, be sure to define --inputs and --outputs. This reduces the enumeration problem complexity considerably.')
    parser.add_argument('--model_path', type=str, default='', help='Relative or absolute path to an SBML model .xml file')
    parser.add_argument('--symbolic', type=str2bool, default=True, help='All computation is done using fractional numbers, and symbolic operations. When disabled, floating point numbers and numeric operations are used (default: True)')
    parser.add_argument('--compress', type=str2bool, default=True, help='Perform compression to which the conversions are invariant, and reduce the network size considerably (default: True)')
    parser.add_argument('--out_path', default='conversion_cone.csv', help='Relative or absolute path to the .csv file you want to save the calculated conversions to')
    parser.add_argument('--add_objective_metabolite', type=str2bool, default=True, help='Add a virtual metabolite containing the stoichiometry of the objective function of the model')
    parser.add_argument('--check_feasibility', type=str2bool, default=False, help='For each found ECM, verify that a feasible flux exists that produces it')
    parser.add_argument('--check_bijection', type=str2bool, default=False, help='Verify completeness of found ECMs by calculating ECMs from EFMs and proving bijection (don\'t use on large networks)')
    parser.add_argument('--print_metabolites', type=str2bool, default=True, help='Print the names and IDs of metabolites in the (compressed) metabolic network')
    parser.add_argument('--print_reactions', type=str2bool, default=True, help='Print the names and IDs of reactions in the (compressed) metabolic network')
    parser.add_argument('--auto_direction', type=str2bool, default=True, help='Automatically determine external metabolites that can only be consumed or produced')
    parser.add_argument('--inputs', type=str, default='', help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that can only be consumed')
    parser.add_argument('--outputs', type=str, default='', help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that can only be produced')
    parser.add_argument('--hide', type=str, default='', help='Comma-separated list of external metabolite indices, as given by --print_metabolites true (before compression), that are transformed into internal metabolites by adding bidirectional exchange reactions')
    args = parser.parse_args()

    if args.model_path == '':
        print('No model given, please specify --model_path')
        exit()

    if len(args.inputs) or len(args.outputs):
        # Disable automatic determination of external metabolite direction if lists are given manually
        args.auto_direction = False

    symbolic = args.symbolic
    model_path = args.model_path

    network = extract_sbml_stoichiometry(model_path, add_objective=args.add_objective_metabolite,
                                         determine_inputs_outputs=args.auto_direction,
                                         skip_external_reactions=True)

    debug_tags = []
    # add_debug_tags(network)

    orig_ids = [m.id for m in network.metabolites]
    orig_N = network.N

    if not args.auto_direction:
        inputs = [int(index) for index in args.inputs.split(',') if len(index)]
        outputs = [int(index) for index in args.outputs.split(',') if len(index)]
        if len(outputs) < 1 and len(inputs) >= 1:
            # If no outputs are given, define all external metabolites that are not inputs as outputs
            print(
                'No output metabolites given or determined from model. All non-input metabolites will be defined as outputs.')
            outputs = np.setdiff1d(network.external_metabolite_indices(), inputs)

        network.set_inputs(inputs)
        network.set_outputs(outputs)

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

    if args.compress:
        network.compress(verbose=True)

    if args.print_reactions and args.compress:
        print('Reactions (after compression):')
        for index, item in enumerate(network.reactions):
            print(index, item.id, item.name, 'reversible' if item.reversible else 'irreversible')

    if args.print_metabolites and args.compress:
        print('Metabolites (after compression):')
        for index, item in enumerate(network.metabolites):
            print(index, item.id, item.name, 'external' if item.is_external else 'internal', item.direction)

    cone = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                               # verbose=True, symbolic=symbolic)
                               input_metabolites=network.input_metabolite_indices(), output_metabolites=network.output_metabolite_indices(), verbose=True, symbolic=symbolic)

    # Undo compression so we have results in the same dimensionality as original data
    expanded_c = np.zeros(shape=(cone.shape[0], len(orig_ids)))

    if args.compress:
        for column, id in enumerate([m.id for m in network.metabolites]):
            orig_column = [index for index, orig_id in enumerate(orig_ids) if orig_id == id][0]
            expanded_c[:, orig_column] = cone[:, column]
    else:
        expanded_c = cone

    np.savetxt(args.out_path, expanded_c, delimiter=',')

    print_ECMs(cone, debug_tags, network, orig_N, args.add_objective_metabolite, args.check_feasibility)

    if args.check_bijection:
        check_bijection(expanded_c, network, model_path, args.add_objective_metabolite)

    end = time()
    print('Ran in %f seconds' % (end - start))
    pass
