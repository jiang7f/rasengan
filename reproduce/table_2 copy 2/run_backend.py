import os
import time
import csv
import signal
import itertools

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from rasengan.problems.facility_location_problem import generate_flp
from rasengan.problems.k_partition_problem import generate_kpp
from rasengan.problems.job_scheduling_problem import generate_jsp
from rasengan.problems.set_cover_problem import generate_scp
from rasengan.problems.graph_coloring_problem import generate_gcp
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import (
    HeaSolver, PenaltySolver, ChocoSolver, RasenganSegmentedSolver,
    AerGpuProvider, AerProvider, DdsimProvider,
)

device = 'cpu'
# device = 'gpu'

num_cases = 10
problem_scale = 4

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)][:problem_scale], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)][:problem_scale], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)][:problem_scale])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)][:problem_scale])

configs_pkg = flp_configs_pkg + kpp_configs_pkg + jsp_configs_pkg + scp_configs_pkg + gcp_configs_pkg
with open(f"table_2.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

depth_and_num_params_csv_path = 'depth_and_num_params.csv'
problems_pkg = flp_problems_pkg + kpp_problems_pkg + jsp_problems_pkg + scp_problems_pkg + gcp_problems_pkg
metrics_lst = ['depth', 'num_params']
solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSegmentedSolver]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, solver, metrics_lst):
    opt = CobylaOptimizer(max_iter=300)
    ddsim = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    baseline_provider = cpu if device == 'cpu' else gpu

    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = baseline_provider if solver in [HeaSolver, PenaltySolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics


if __name__ == '__main__':
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    with open(f'{depth_and_num_params_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
        futures = []
        for solver in solvers:
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    num_layers = 5
                    future = executor.submit(process_layer, problem, num_layers, solver, metrics_lst)
                    futures.append((future, pkid, solver.__name__, num_layers))

        start_time = time.perf_counter()
        for future, pkid, solver, num_layers in futures:
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend(result)
                print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
            finally:
                row = [pkid, solver, num_layers] + diff
                with open(f'{depth_and_num_params_csv_path}', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    print(f'Data has been written to {depth_and_num_params_csv_path}')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)

evaluate_csv_path = 'evaluate.csv'

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(kpp_problems_pkg),
        enumerate(jsp_problems_pkg),
        enumerate(scp_problems_pkg),
        enumerate(gcp_problems_pkg),
    )
)

solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSegmentedSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider(transpile_mode=0)
    cpu = AerProvider()
    gpu = AerGpuProvider()
    baseline_provider = cpu if device == 'cpu' else gpu

    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = baseline_provider if solver in [HeaSolver, PenaltySolver] else ddsim,
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
    print(evaluate_csv_path)
    with open(f'{evaluate_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:
            # 防止GPU内存溢出
            if device == 'gpu' and solver in [HeaSolver, PenaltySolver]:
                num_processes = 2**(4 - diff_level)
            else:
                num_processes = num_processes_cpu // 4

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

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
                        with open(f'{evaluate_csv_path}', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {evaluate_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")