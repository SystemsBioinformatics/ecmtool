import copy
import pandas as pd
from ecmtool import extract_sbml_stoichiometry, get_conversion_cone
from ecmtool.helpers import unsplit_metabolites, print_ecms_direct, unique
import numpy as np
import os


def calc_ECMs(file_path, print_results=False, input_file_path=''):
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
    metab_info_ext = [(ind, metab.id, metab.name, metab.direction) for ind, metab in
                      enumerate(network.metabolites) if metab.is_external]

    """I extract some information about the external metabolites for checking"""
    metab_info_ext_df = pd.DataFrame(metab_info_ext, columns=['metab_ind', 'metab_id', 'metab_name', 'Direction'])

    """You can choose to save this information, by uncommenting this line"""
    #    metab_info_ext_df.to_csv(path_or_buf='external_info_iJR904.csv', index=False)

    """If an input file is supplied, we set in input, output, and hide metabolites from this"""
    if input_file_path:  # If no input file is supplied, standard detection of ecmtool is used
        info_metabs_df = pd.read_csv(input_file_path)
        info_metabs_input = info_metabs_df[info_metabs_df.Input == 1]
        info_metabs_output = info_metabs_df[info_metabs_df.Output == 1]
        info_metabs_hidden = info_metabs_df[info_metabs_df.Hidden == 1]

        # Get the indices that correspond to the metabolites that are inputs, outputs, or hidden.
        input_inds = list(info_metabs_input.Index.values)
        output_inds = list(info_metabs_output.Index.values) + [ind for ind, metab in enumerate(network.metabolites) if
                                                               metab.id == 'objective']
        hide_inds = list(info_metabs_hidden.Index.values)
        prohibit_inds = [ind for ind, metab in enumerate(network.metabolites) if
                         (metab.is_external) & (not ind in input_inds + output_inds + hide_inds) & (
                             not metab.id == 'objective')]
        both_inds = [ind for ind in range(len(network.metabolites)) if (ind in input_inds) and (ind in output_inds)]

        # Use input information to set input, output, hidden, and prohibited metabolites
        network.set_inputs(input_inds)
        network.set_outputs(output_inds)
        network.set_both(both_inds)
        network.prohibit(prohibit_inds)
        network.hide(hide_inds)

        # Print comma-separated lists of input information. These lists can be used for running the same computation
        # via command line, for example to use other arguments
        print(','.join(map(str, input_inds)))
        print(','.join(map(str, output_inds)))
        print(','.join(map(str, hide_inds)))
        print(','.join(map(str, prohibit_inds)))

    """Keep a copy of the full network before compression. This can be nice for later."""
    full_network = copy.deepcopy(network)
    orig_N = network.N

    """"Split in and out metabolites, to facilitate ECM computation"""
    network.split_in_out(only_rays=False)

    """Stap 2: compress network"""
    network.compress(verbose=True)

    """Stap 3: Ecms enumereren"""
    #  In this script, indirect intersection is used. Use command line options to use direct intersection
    cone = get_conversion_cone(network.N, network.external_metabolite_indices(), network.reversible_reaction_indices(),
                            network.input_metabolite_indices(), network.output_metabolite_indices(), only_rays=False,
                            verbose=True)
    cone_transpose, ids = unsplit_metabolites(np.transpose(cone), network)
    cone = np.transpose(cone_transpose)

    if print_results:
        print_ecms_direct(np.transpose(cone), ids)

    cone = cone.transpose()  # columns will be the different ECMs, rows are metabolites

    return cone, ids, full_network


def print_ECMs(cone, debug_tags, network, orig_N, add_objective_metabolite):
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


def get_efms(N, reversibility, verbose=True, efmtool_path=os.getcwd()):
    """
    Uses Matlab and EFMtool (https://csb.ethz.ch/tools/software/efmtool.html) to calculate the EFMs of a model.
    :param N: stoichiometry matrix
    :param reversibility: list of booleans indicating if reaction is reversible
    :param verbose:
    :param efmtool_path: Path to where EFM software is installed on machine
    :return: matrix with efms
    """
    import matlab.engine
    engine = matlab.engine.start_matlab()
    engine.cd(efmtool_path)
    result = engine.CalculateFluxModes(matlab.double([list(row) for row in N]), matlab.logical(reversibility))
    if verbose:
        print('Fetching calculated EFMs')
    size = result['efms'].size
    shape = size[1], size[0] # _data is in transposed form w.r.t. the result matrix
    efms = np.reshape(np.array(result['efms']._data), shape)
    if verbose:
        print('Finishing fetching calculated EFMs')
    return efms


def check_bijection_csvs(ecms_first_df, ecms_second_df):
    """
    :param ecms_first_df: DataFrame
            Matrix with ecms as rows, metabolites as cols
            colname should give metab_id
    :param ecms_second_df: DataFrame
            Matrix with ecms as rows, metabolites as cols
            colname should give metab_id
    :return bijection_YN: Boolean
    :return ecms_second_min_ecms_first: np.array
            Matrix with as columns the ECMs that were in the second but not in the first set
    :return ecms_first_min_ecms_second: np.array
            Matrix with as columns the ECMs that were in the first but not in the second set
    """
    # We first remove duplicates from both
    metab_ids_first = list(ecms_first_df.columns)
    metab_ids_second = list(ecms_second_df.columns)
    ecms_first = np.transpose(ecms_first_df.values)
    ecms_second = np.transpose(ecms_second_df.values)
    n_ecms_first_non_unique = ecms_first.shape[1]
    n_ecms_second_non_unique = ecms_second.shape[1]
    ecms_first = np.transpose(unique(np.transpose(ecms_first)))
    ecms_second = np.transpose(unique(np.transpose(ecms_second)))
    n_ecms_first = ecms_first.shape[1]
    n_ecms_second = ecms_second.shape[1]

    # Find matching of metab_ids
    matching_inds =np.zeros(len(metab_ids_first))
    for id_ind, id in enumerate(metab_ids_first):
        matching_inds[id_ind] = [id_ind_sec for id_ind_sec, id_sec in enumerate(metab_ids_second) if id_sec == id][0]

    # Make sure that second ecms metabolites are in the same order
    matching_inds = matching_inds.astype(int)
    ecms_second = ecms_second[matching_inds, :]

    if n_ecms_first_non_unique - n_ecms_first > 0:
        print("Watch out. The first set of ECMs has duplicates")
    if n_ecms_second_non_unique - n_ecms_second > 0:
        print("Watch out. The second set of ECMs has duplicates")

    # Normalize both sets of ECMs
    sum_columns_first = np.sum(np.abs(ecms_first), axis=0)
    sum_columns_first = sum_columns_first[np.newaxis,:]
    ecms_first = ecms_first / np.repeat(sum_columns_first, ecms_first.shape[0], axis=0)

    sum_columns_second = np.sum(np.abs(ecms_second), axis=0)
    sum_columns_second = sum_columns_second[np.newaxis,:]
    ecms_second = ecms_second / np.repeat(sum_columns_second, ecms_second.shape[0], axis=0)

    found_match_ecms_first = [False] * n_ecms_first
    no_match_ecms_second = list(range(n_ecms_second))
    for ecm_first_ind in range(n_ecms_first):
        if ecm_first_ind % 100 == 0:
            print('%d/%d ECMs checked for matches' % (ecm_first_ind, n_ecms_first))
        ecm_first = ecms_first[:, ecm_first_ind]
        for index, ecm_second_ind in enumerate(no_match_ecms_second):
            ecm_second = ecms_second[:, ecm_second_ind]

            if max(ecm_first - ecm_second) <= 10 ** -3:
                found_match_ecms_first[ecm_first_ind] = True
                del no_match_ecms_second[index]
                break

    ecms_first_min_ecms_second_inds = np.where([not found for found in found_match_ecms_first])[0]
    ecms_second_min_ecms_first_inds = no_match_ecms_second

    ecms_first_min_ecms_second = ecms_first[:, ecms_first_min_ecms_second_inds]
    ecms_second_min_ecms_first = ecms_second[:, ecms_second_min_ecms_first_inds]

    if not (ecms_first_min_ecms_second.shape[1] > 0 or ecms_second_min_ecms_first.shape[1] > 0):
        bijection_YN = True
    else:
        bijection_YN = False

    return bijection_YN, ecms_first_min_ecms_second, ecms_second_min_ecms_first