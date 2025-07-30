import os
import time
import csv
import signal
import random
import itertools

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from rasengan.problems.facility_location_problem import generate_flp
import numpy as np
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import (
    PenaltySolver, ChocoSolver, RasenganSegmentedSolver,
    AerGpuProvider, AerProvider, DdsimProvider,
)

np.random.seed(0x7f)
random.seed(0x7f)

num_cases = 10

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2)], 1, 10)

configs_pkg = flp_configs_pkg
with open(f"figure_9.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

csv_path = "more_layers_qaoa.csv"

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

solvers = [ChocoSolver, PenaltySolver, RasenganSegmentedSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics
num_layers_list = range(1, 21, 1)

def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=1000)
    aer = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        # provider = gpu if solver in [HeaSolver, PenaltySolver] else aer,
        provider = cpu if solver in [PenaltySolver] else aer,
        num_layers = num_layers,
        shots = 1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    with open(f'{csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for layer in num_layers_list:
            for solver in solvers:
                num_processes = 100

                if solver == RasenganSegmentedSolver and layer > 1:
                    continue

                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    # layer = 5
                    for pbid, prb in enumerate(problems):
                        print(f'{pkid}-{pbid}, {layer}, {solver} build')
                        future = executor.submit(process_layer, prb, layer, solver)
                        futures.append((future, prb, pkid, pbid, layer, solver.__name__))

                    start_time = time.perf_counter()
                    for future, prb, pkid, pbid, layer, solver in futures:
                        current_time = time.perf_counter()
                        remaining_time = max(set_timeout - (current_time - start_time), 0)
                        diff = []
                        try:
                            metrics = future.result(timeout=remaining_time)
                            diff.extend(metrics)
                            print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
                        except MemoryError:
                            print(f"Task for problem {pkid}-{pbid} L={layer} {solver} encountered a MemoryError.")
                            for dict_term in evaluation_metrics:
                                diff.append('memory_error')
                        except TimeoutError:
                            print(f"Task for problem {pkid}-{pbid} L={layer} {solver} timed out.")
                            for dict_term in evaluation_metrics:
                                diff.append('timeout')
                        except Exception as e:
                            print(f"An error occurred: {e}")
                        finally:
                            row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver] + diff
                            with open(f'{csv_path}', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(row)  # Write row immediately
                            num_complete += 1
                            if num_complete == len(futures):
                                print(f'problem_pkg_{pkid} has finished')
                                for process in executor._processes.values():
                                    os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")