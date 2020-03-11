from fractions import Fraction

import numpy as np
from ecmtool.helpers import to_fractions


def check_bijection_Erik(ecms_first, ecms_second, network):
    """
    :param ecms_first: np.array
            Matrix with ecms as columns, metabolites as rows
    :param ecms_second: np.array
            Matrix with ecms as columns, metabolites as rows
    :return bijection_YN: Boolean
    :return ecms_second_min_ecms_first: np.array
            Matrix with as columns the ECMs that were in the second but not in the first set
    :return ecms_first_min_ecms_second: np.array
            Matrix with as columns the ECMs that were in the first but not in the second set
    :param network: ecmtool.network class
            ecmtool.network object as comes from ECMtool and which is used for calculating ecms_matrix
    """
    # We first remove duplicates from both
    n_ecms_first_non_unique = ecms_first.shape[1]
    n_ecms_second_non_unique = ecms_second.shape[1]
    ecms_first = np.transpose(unique_Erik(np.transpose(ecms_first)))
    ecms_second = np.transpose(unique_Erik(np.transpose(ecms_second)))
    n_ecms_first = ecms_first.shape[1]
    n_ecms_second = ecms_second.shape[1]

    if n_ecms_first_non_unique - n_ecms_first > 0:
        print("Watch out. The first set of ECMs has duplicates")
    if n_ecms_second_non_unique - n_ecms_second > 0:
        print("Watch out. The second set of ECMs has duplicates")

    # Normalize both sets of ECMs
    normalization_order = determine_normalization_order(ecms_first, network)
    ecms_first = normalize_ECMS_Erik(ecms_first, network, normalization_order=normalization_order)
    ecms_second = normalize_ECMS_Erik(ecms_second, network, normalization_order=normalization_order)

    found_match_ecms_first = [False] * n_ecms_first
    no_match_ecms_second = list(range(n_ecms_second))
    for ecm_first_ind in range(n_ecms_first):
        if ecm_first_ind % 100 == 0:
            print('%d/%d ECMs checked for matches' % (ecm_first_ind, n_ecms_first))
        ecm_first = ecms_first[:, ecm_first_ind]
        for index, ecm_second_ind in enumerate(no_match_ecms_second):
            ecm_second = ecms_second[:, ecm_second_ind]

            if max(ecm_first - ecm_second) <= 10 ** -6:
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


def check_bijection_csvs(ecms_first, ecms_second):
    """
    :param ecms_first: np.array
            Matrix with ecms as columns, metabolites as rows
    :param ecms_second: np.array
            Matrix with ecms as columns, metabolites as rows
    :return bijection_YN: Boolean
    :return ecms_second_min_ecms_first: np.array
            Matrix with as columns the ECMs that were in the second but not in the first set
    :return ecms_first_min_ecms_second: np.array
            Matrix with as columns the ECMs that were in the first but not in the second set
    """
    # We first remove duplicates from both
    n_ecms_first_non_unique = ecms_first.shape[1]
    n_ecms_second_non_unique = ecms_second.shape[1]
    ecms_first = np.transpose(unique_Erik(np.transpose(ecms_first)))
    ecms_second = np.transpose(unique_Erik(np.transpose(ecms_second)))
    n_ecms_first = ecms_first.shape[1]
    n_ecms_second = ecms_second.shape[1]

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

            if max(ecm_first - ecm_second) <= 10 ** -6:
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


def determine_normalization_order(ecms_matrix, network):
    """
    Determine order of metabolites to which we are going to normalize.
    :return normalization_order: list of metab-IDs
            List of ordered metabolite-IDs
    :param ecms_matrix:
            This array contains the ECMs as columns and the metabolites as rows
    :param network: ecmtool.network class
            ecmtool.network object as comes from ECMtool and which is used for calculating ecms_matrix
    """
    metabs = []
    metabs_usage = []
    for metab_ind, metab in enumerate(network.metabolites):
        metabs.append(metab.id)  # Store all other metabolite ids
        metabs_usage.append(np.count_nonzero(ecms_matrix[metab_ind, :]))  # And their number of occurrences in ECMs

    # Order the tertiary objectives for how often they are used
    normalization_order = [x for (y, x) in
                           sorted(zip(metabs_usage, metabs), key=lambda pair: pair[0],
                                  reverse=True)]

    return normalization_order


def normalize_ECMS_Erik(ecms_matrix, network, normalization_order=[]):
    """
    Normalizes ECMs first to first objective. If there are also lower bounds that act as a kind of second objective.
    Then normalize the ECMs with zero objective to this second objective.
    :return ecms_matrix: np.array
            This array contains the normalized ECMs as columns and the metabolites as rows
    :param ecms_matrix:
            This array contains the ECMs as columns and the metabolites as rows
    :param network: Network class
            Network class as comes from ECMtool
    :param normalization_order: list of metab-IDs
            List of ordered metabolite-IDs
    """
    # Determine an order for normalizing the metabolites, if none is given
    # ECMs that are used often are picked first for normalization. To compare two ECM results, make sure to pick the
    # same normalization order
    if not len(normalization_order):
        normalization_order = determine_normalization_order(ecms_matrix, network)

    not_normalized_yet = list(range(ecms_matrix.shape[1]))  # This tracks which ECMs need to be normalized still

    # Then normalize all ECMs to one of the metabolites
    for metab in normalization_order:
        print('Normalizing ' + metab)
        if not len(not_normalized_yet):
            break

        metab_index = [index for index, met in enumerate(network.metabolites) if met.id == metab][0]
        ecms_matrix, not_normalized_yet = normalize_to_row_Erik(ecms_matrix, metab_index, not_normalized_yet)

    return ecms_matrix


def normalize_to_row_Erik(matrix, row_ind, not_normalized_yet):
    """
    :param matrix: np.array
            Matrix that should be normalized
    :param row_ind: int
            Row that should be normalized to
    :param not_normalized_yet: list of ints
            List of column indices that still need normalization
    :return: matrix: np.array
            Normalized matrix
    :return: not_normalized_yet: list of ints
            Updated list of column indices that still need normalization
    """
    obj_row = matrix[row_ind, :]
    div_factors = [Fraction(1, 1)] * matrix.shape[1]  # By default, divide by 1
    normalized_indices = []

    # Find those colunns that are not yet normalized, but can be normalized using this row
    for col_ind, ecm_ind in enumerate(not_normalized_yet):
        if obj_row[ecm_ind] != 0:
            div_factors[ecm_ind] = obj_row[ecm_ind]  # If column can be normalized, divide by the obj_row-value
            normalized_indices.append(col_ind)

    not_normalized_yet = np.delete(not_normalized_yet, normalized_indices)

    divisor_matrix = np.tile(div_factors, (matrix.shape[0], 1))
    matrix = np.divide(matrix, divisor_matrix)

    return matrix, not_normalized_yet


def unique_Erik(matrix):
    # Tom made this function and seems to use some set properties
    unique_set = {tuple(row) for row in matrix if np.count_nonzero(row) > 0}
    return np.vstack(unique_set) if len(unique_set) else to_fractions(np.ndarray(shape=(0, matrix.shape[1])))
