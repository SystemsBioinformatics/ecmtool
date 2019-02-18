from multiprocessing import Process, Queue, cpu_count

import matplotlib.pyplot as plt
from matplotlib import interactive

from ecmtool.helpers import *
from models.model6 import *

# Build P and M matrices (eq. 10)
P, M = get_P_M(metabolites, metabolic_reactions, enzyme_reactions)

# Calculate a and b net volume rates (bottom pg. 3)
net_volume_a, net_volume_b = get_net_volumes(metabolites, metabolic_reactions, metabolite_molar_volumes,
                                             enzyme_reactions, enzyme_molar_volumes, P, M)


def get_results_from_growth_rate(worker_num, start_rate, end_rate, steps, queue):
    active_rows = []
    active_cols = []

    for i, cur_growth_rate in enumerate(np.linspace(start_rate, end_rate, steps)):
        # Build A matrix (eq. 19)
        A = build_A_matrix(metabolites, metabolic_reactions, metabolite_concentrations, P, M, net_volume_a, net_volume_b,
                           rate_functions, cur_growth_rate)

        print('Worker #%d calculating growth rate %f (%d%%)' % (
        (worker_num + 1), cur_growth_rate, int((i / float(steps)) * 100)))

        if not len(active_rows) or not len(active_cols):
            # Run polco to enumerate extreme rays
            for ray_index, result in enumerate(get_extreme_rays(A)):
                result = np.append([ray_index, cur_growth_rate], result)
                if result[-1] == 0.0:
                    print('Worker #%d: Got invalid extreme ray result -> ' % (worker_num + 1), str(result))
                    result[2:] = [0] * (len(result) - 2)
                else:
                    columns, rows = get_used_rows_columns(A, result)
                    active_rows.append(rows)
                    active_cols.append(columns)

                queue.put(result)
        else:
            # One existing set of EGMs is already present, we can take a shortcut from here
            for EGM_index in range(len(active_rows)):
                A_red = build_reduced_A_matrix(active_rows[EGM_index], active_cols[EGM_index], metabolite_concentrations,
                                               P, M, net_volume_a, net_volume_b, rate_functions, cur_growth_rate)
                b = np.zeros(shape=(A_red.shape[0], 1))
                b[-1] = cur_growth_rate
                interim_result = np.linalg.solve(A_red, b)
                result = np.zeros(shape=(1, A.shape[1]))

                for index, column in enumerate(active_cols[EGM_index]):
                    result[0, column] = interim_result[index, :]

                result[0, -1] = 1

                result = np.append([cur_growth_rate, EGM_index], result)
                queue.put(result)



def result_writer(queue, workers, metabolic_reactions, rate_functions, metabolite_concentrations):
    with open('results.csv', 'w') as file:
        header = '"growth_rate", "EGM_index", %s, %s, %s, %s, %s, %s\n' % (', '.join(
            ['"B%s"' % str(i) for i in range(1, len(metabolic_reactions) + 2)]), '"ribosome_concentration"', ', '.join(
            ['"e%s"' % str(i) for i in range(1, len(metabolic_reactions) + 1)]), ', '.join(
            ['"V%s"' % str(i) for i in range(1, len(metabolic_reactions) + 1)]), ', '.join(
            ['"W%s"' % str(i) for i in range(1, len(metabolic_reactions) + 2)]), ', '.join(
            ['"MW%s"' % str(i) for i in range(1, len(metabolic_reactions))]))
        file.write(header)

        print('Writer thread listening for results')

        while True:
            workers_alive = [worker.is_alive() for worker in workers]
            if True not in workers_alive:
                break

            result = queue.get()

            # If the ribosomal beta value is 0, the EGM is invalid and should have
            # all zero values written to file
            if result[-1] != 0.0:
                # Normalise by ribosomal beta
                result = normalise_betas(result)
                # Add computed reaction rates
                result = add_rates(result, rate_functions, metabolite_concentrations, net_volume_a, net_volume_b, M)
            else:
                result = np.append(result, [0] * (len(metabolic_reactions) * 3 + 1), axis=0) # Add dummy rib_conc, Bi, Vi, Wi values

            file.write(', '.join([str(x) for x in result]) + '\n')
            file.flush()


def result_plotter(queue, workers, metabolic_reactions, rate_functions, metabolite_concentrations):
    datatypes = {
        'B': {
            'start': 2,
            'length': len(metabolic_reactions) + 1
        },
        'e': {
            'start': 2 + len(metabolic_reactions) + 1,
            'length': len(metabolic_reactions)
        },
        'v': {
            'start': 2 + 2*len(metabolic_reactions) + 1,
            'length': len(metabolic_reactions)
        },
        'w': {
            'start': 2 + 3*len(metabolic_reactions) + 1,
            'length': len(metabolic_reactions) + 1
        },
        'Mw': {
            'start': 2 + 4*len(metabolic_reactions) + 2,
            'length': len(metabolic_reactions)
        }
    }

    # Initialise figures
    for index, datatype_key in enumerate(datatypes.keys()):
        datatype = datatypes[datatype_key]
        plt.figure(index + 1)
        plt.title(datatype)
    interactive(True)

    while True:
        result = queue.get()

        workers_alive = [worker.is_alive() for worker in workers]
        if True not in workers_alive and queue.empty():
            interactive(False)
            plt.show()
            break

        # If the ribosomal beta value is 0, the EGM is invalid and should have
        # all zero values written to file
        if result[-1] != 0.0:
            # Normalise by ribosomal beta
            result = normalise_betas(result)
            # Add computed reaction rates
            result = add_rates(result, rate_functions, metabolite_concentrations, net_volume_a, net_volume_b, M)
        else:
            result = np.append(result, [0] * (len(metabolic_reactions) * 4 + 2),
                               axis=0)  # Add dummy rib_conc, Bi, Vi, Wi values

        growth_rate = result[0]
        EGM_index = int(result[1])
        colors = ['r', 'g', 'b', 'y']

        for index, datatype_key in enumerate(datatypes.keys()):
            datatype = datatypes[datatype_key]
            fig = plt.figure(index+1)
            for item_number in range(datatype['length']):
                data_start = datatype['start'] + item_number
                plt.subplot(100 * datatype['length'] + 11 + item_number)
                plt.title('%s%d' % (datatype_key, item_number))
                plt.scatter([growth_rate], [result[data_start]], c=colors[EGM_index], label='EGM %d' % EGM_index)

        # Update figures
        plt.pause(0.0001)


def launch_workers(start, end, steps):
    queue = Queue()
    num_cpus = cpu_count()
    steps_per_worker = steps / num_cpus
    step_size = (float(end) - start) / steps
    workers = []

    for worker_num in range(num_cpus):
        print('Launching worker #%d' % (worker_num + 1))
        start_rate = start + (step_size * steps_per_worker * worker_num)
        end_rate = start_rate + (step_size * steps_per_worker)
        worker_proc = Process(target=get_results_from_growth_rate,
                              args=(worker_num, start_rate, end_rate, steps_per_worker, queue))
        worker_proc.daemon = True
        worker_proc.start()
        workers.append(worker_proc)

    print('Running result processor')
    current_target = result_plotter
    current_target(queue, workers, metabolic_reactions, rate_functions, metabolite_concentrations)


if __name__ == '__main__':
    launch_workers(10**-6, max_growth_rate, 100)
    print('Done, exiting')
    pass
