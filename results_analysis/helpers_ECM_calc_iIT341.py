import copy

import pandas as pd
from ecmtool import extract_sbml_stoichiometry, get_conversion_cone
from scipy.optimize import linprog


def calc_ECMs(file_path, print_results=False, hide_metabs=[], input_metabs=[], output_metabs=[],
              both_metabs = [], external_metabs=[]):
    """
    Calculates ECMs using ECMtool
    :return ecms: np.array
            This array contains the ECMs as columns and the metabolites as rows
    :param file_path: string
            String with path to the SBML-file.
    :param reactions_to_tag: list with strings
            List with reaction-IDs of reactions that need to be tagged
    :param print_results: Boolean
    :param hide_metabs: indices of metabolites that should be ignored
    """
    # Stap 1: netwerk bouwen
    network = extract_sbml_stoichiometry(file_path, determine_inputs_outputs=True)

    external_inds = [ind for ind, metab in enumerate(network.metabolites) if metab.is_external]

    """The following are just for checking the inputs to this program."""
    metab_info_ext = [(ind, metab.id, metab.direction) for ind, metab in
                      enumerate(network.metabolites) if metab.is_external]

    """I extract some information about the external metabolites for checking"""
    metab_info_ext_df = pd.DataFrame(metab_info_ext, columns=['metab_ind','metab_id', 'Direction'])

    """You can choose to save this information, but I use the same file for inputting information, so it is not very practical at the moment."""
#    metab_info_ext_df.to_csv(path_or_buf='external_info_iJR904.csv', index=False)

    """Read in input, output, and hide information"""
    # info_metabs_df = pd.read_csv('external_info_iJR904.csv')
    # hide_metabs = [metab for ind, metab in enumerate(info_metabs_df['metab_ind']) if info_metabs_df['hideYN'][ind]]

    """Get the indices that correspond to the metabolites that need to be hidden, and then hide them."""
    # if len(hide_metabs) > 0:
    #    network.hide(hide_metabs)

    """Keep a copy of the full network before compression. This can be nice for later."""
    full_network = copy.deepcopy(network)
    orig_N = network.N

    """Stap 2: compress network"""
    network.compress(verbose=True)

    """Stap 3: Ecms enumereren"""
    cone = network.uncompress(
        get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                            network.input_metabolite_indices(), network.output_metabolite_indices(), only_rays=False,
                            verbose=True))

    if print_results:
        indices_to_tag = []
        print_ECMs(cone, indices_to_tag, network, orig_N, add_objective_metabolite=True, check_feasibility=False)

    cone = cone.transpose()  # columns will be the different ECMs, rows are metabolites

    return cone, full_network


def print_ECMs(cone, debug_tags, network, orig_N, add_objective_metabolite, check_feasibility):
    for index, ecm in enumerate(cone):
        # Normalise by objective metabolite, if applicable
        objective_index = -1 - len(debug_tags)
        objective = ecm[objective_index]
        if add_objective_metabolite and objective > 0:
            ecm /= objective

        metabolite_ids = [met.id for met in
                          network.metabolites] if not network.compressed else network.uncompressed_metabolite_ids

        print('\nECM #%d:' % index)
        for metabolite_index, stoichiometry_val in enumerate(ecm):
            if stoichiometry_val != 0.0:
                print('%s\t\t->\t%.4f' % (metabolite_ids[metabolite_index], stoichiometry_val))

        if check_feasibility:
            allowed_error = 10 ** -6
            solution = linprog(c=[1] * orig_N.shape[1], A_eq=orig_N, b_eq=cone[index, :],
                               bounds=[(-1000, 1000)] * orig_N.shape[1], options={'tol': allowed_error})
            print('ECM satisfies stoichiometry' if solution.status == 0 else 'ECM does not satisfy stoichiometry')
