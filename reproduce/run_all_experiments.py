import os
import csv
import json
import time
import signal
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import norm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from rasengan.problems.facility_location_problem import generate_flp
from rasengan.problems.k_partition_problem import generate_kpp
from rasengan.problems.job_scheduling_problem import generate_jsp
from rasengan.problems.set_cover_problem import generate_scp
from rasengan.problems.graph_coloring_problem import generate_gcp
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import (
    HeaSolver, PenaltySolver, ChocoSolver, RasenganSolver, RasenganSegmentedSolver, 
    QtoSolver, AerProvider, AerGpuProvider, DdsimProvider, NoisyDdsimProvider,
    FakeQuebecProvider, FakeKyivProvider, FakeBrisbaneProvider, BitFlipNoiseAerProvider,
)
from rasengan.solvers.qiskit.explorer import QtoExplorer
from rasengan.utils.qiskit_plugin import detect_device

num_cases = 10

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2)], 1, 10)

csv_path = "figure_9/more_layers_qaoa.csv"

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

solvers = [ChocoSolver, PenaltySolver, RasenganSolver]
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
        for layer in tqdm(num_layers_list, desc="Evaluating across num_layers"):
            for solver in solvers:
                num_processes = 100

                if solver == RasenganSolver and layer > 1:
                    continue

                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    # layer = 5
                    for pbid, prb in enumerate(problems):
                        # print(f'{pkid}-{pbid}, {layer}, {solver} build')
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
                            # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                                # print(f'problem_pkg_{pkid} has finished')
                                for process in executor._processes.values():
                                    os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")



df = pd.read_csv(csv_path)

rasengan_arg = df.query("method == 'RasenganSolver' and layers == 1")['ARG'].mean()

grouped_df = df.groupby(['pkid', 'layers', 'variables', 'constraints', 'method'], as_index=False).agg({
    'iteration_count':'mean',
    'classcial':'mean',
    'run_times':'mean',
    "ARG": 'mean',
    'in_constraints_probs': 'mean',
    'best_solution_probs': 'mean',
})

pivot_df = grouped_df.pivot(index =['pkid', 'layers', 'variables', 'constraints'], columns='method', values=["ARG", 'best_solution_probs', 'classcial', 'run_times','iteration_count'])
method_order = ['PenaltySolver', 'ChocoSolver', 'RasenganSolver']
pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product([["ARG", 'best_solution_probs', 'run_times', 'iteration_count', 'classcial'], method_order]))



df = pd.read_csv(csv_path)
grouped_df = df.groupby(['pkid', 'layers', 'variables', 'constraints', 'method'], as_index=False).agg({
    'ARG': 'mean'
})
rasengan_arg = df.query("method == 'RasenganSolver' and layers == 1")['ARG'].mean()

scale = 1
fig = plt.figure(figsize=(22*scale, 12*scale))
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    "lines.markersize": 35,
    "lines.markeredgewidth": 4,
    "markers.fillstyle": 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

ax = plt.axes((0, 0, 1, 0.6))

colors = ['#6A9C89', '#B8001F']
labels = ['P-QAOA', 'Choco-Q']

for idx, method in enumerate(['PenaltySolver', 'ChocoSolver']):
    data = grouped_df[grouped_df.method == method]
    x = np.arange(len(data.pkid))
    y = data['ARG'].to_list()
    ax.plot(x + idx * 0.2, y, marker='o', markersize=25, color=colors[idx], label=labels[idx],
            linestyle='-', linewidth=4, markeredgewidth=3, markerfacecolor=colors[idx], markeredgecolor='black')

ax.axhline(y=rasengan_arg, color='orange', linestyle='--', linewidth=3, label='Rasengan')

ax.grid(True, linestyle='--', linewidth=1.5, axis='y')

plt.yscale('log')
plt.xlabel('#layers')
plt.ylabel('ARG')
plt.legend(loc='upper left', ncol=3, frameon=False, bbox_to_anchor=(0, 1.1, 1, 0.2), mode="expand", borderaxespad=0)

title = 'Figure 9: Evaluation of ARG using the different number of QAOA layers'
plt.suptitle(title, y=-0.2, fontsize=48)
plt.savefig(f'figure_9/{title}.svg', bbox_inches='tight')

print("Figuire 9 finished.")

m, n = 7, 7
num_cases_1 = 1
scale_list = [(i, j) for i in range(1, m + 1) for j in range(2, n + 1)][:22]
a, b = generate_flp(num_cases_1, scale_list, 1, 20)

# for fig (c) and (d)
num_cases_2 = 10
flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases_2, [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)], 1, 100)

csv_path = "figure_10/segments_and_depth.csv"
num_processes_cpu = os.cpu_count()
num_processes = max(1, num_processes_cpu // 4)

metrics_lst = ['depth', 'num_params']

with open(f'{csv_path}', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "variables"] + metrics_lst + ['depth_2', 'num_params_2'])

def process_layer(i):
    scale = scale_list[i]
    prb_model = a[i][0]
    num_vars = b[i][0][1]

    prb_model.set_penalty_lambda(200)

    opt = CobylaOptimizer(max_iter=50)
    provider = FakeQuebecProvider(transpile_mode=1)

    solver_1 = RasenganSolver(
        prb_model=prb_model,
        optimizer=opt,
        provider=provider,
        num_layers=5,
        shots=1024,
        opt_mode=[0, 0]
    )
    solver_2 = QtoSolver(
        prb_model=prb_model,
        optimizer=opt,
        provider=provider,
        num_layers=5,
        shots=1024,
    )

    metrics_1 = solver_1.circuit_analyze(metrics_lst)
    metrics_2 = solver_2.circuit_analyze(metrics_lst)

    return [scale[0], scale[1], num_vars] + metrics_1 + metrics_2

futures = []
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    for i in range(len(scale_list)):
        futures.append(executor.submit(process_layer, i))

    for future in tqdm(futures, desc="Evaluating #segments and depth"):
        result = future.result()
        with open(f'{csv_path}', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(result)

print(f'Data has been written to {csv_path}')

large_evaluation_csv_path = "figure_10/large_evaluation.csv"
problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)
solvers = [ChocoSolver, RasenganSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics


def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=500)
    aer = DdsimProvider()
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = aer,
        num_layers = num_layers,
        shots = 1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

if __name__ == '__main__':
    print("Evaluating ARG on large-scale problems:")
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    # print(evaluate_csv_path)
    with open(f'{large_evaluation_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    num_processes = num_processes_cpu // 4
    # pkid-pbid: 问题包序-包内序号
    
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:

            solver_name = solver.__name__
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

                for pbid, prb in enumerate(problems):
                    future = executor.submit(process_layer, prb, layer, solver)
                    futures.append((future, prb, pkid, pbid, layer, solver_name))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver in tqdm(futures, desc=f"problem_{pkid} using {solver_name}"):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                        with open(f'{large_evaluation_csv_path}', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {large_evaluation_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

num_cases_2 = 5

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases_2, [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)], 1, 100)

noisy_evaluation_csv_path = "figure_10/noisy_evaluation.csv"
problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)
solvers = [RasenganSegmentedSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics


def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=500)
    aer = NoisyDdsimProvider()
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = aer,
        num_layers = num_layers,
        shots = 1024,
        num_segments = 10,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

if __name__ == '__main__':
    print("Evaluating ARG under noise:")
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    # print(evaluate_csv_path)
    with open(f'{noisy_evaluation_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    num_processes = num_processes_cpu // 4
    # pkid-pbid: 问题包序-包内序号
    
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:

            solver_name = solver.__name__
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

                for pbid, prb in enumerate(problems):
                    future = executor.submit(process_layer, prb, layer, solver)
                    futures.append((future, prb, pkid, pbid, layer, solver_name))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver in tqdm(futures, desc=f"problem_{pkid} using {solver_name}"):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                        with open(f'{noisy_evaluation_csv_path}', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {noisy_evaluation_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

df = pd.read_csv(csv_path)
df_sorted = df.sort_values(by='variables')

def preprocess_df(filepath, methods):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['pbid'])
    df = df[df['ARG'] <= 100000]
    grouped = df.groupby(['variables', 'method'], as_index=False).agg({
        'iteration_count': 'mean',
        'classcial': 'mean',
        'run_times': 'mean',
        'ARG': 'mean',
        'in_constraints_probs': 'mean',
        'best_solution_probs': 'mean',
    })
    values = ["variables", "ARG", 'best_solution_probs', 'classcial', 'in_constraints_probs', 'run_times', 'iteration_count']
    pivot = grouped.pivot(index='variables', columns='method', values=values)
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([values, methods]))
    return pivot

pivot_df_1 = preprocess_df(large_evaluation_csv_path, methods=['ChocoSolver', 'RasenganSolver'])
pivot_df_2 = preprocess_df(noisy_evaluation_csv_path, methods=['RasenganSegmentedSolver'])


mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    "lines.markersize": 35,
    "lines.markeredgewidth": 4,
    "markers.fillstyle": 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

fig = plt.figure(figsize=(22, 12))

ax1 = plt.axes([0, 0.7, 0.4, 0.5])
ax2 = plt.axes([0.5, 0.7, 0.5, 0.5])
ax3 = plt.axes([0, 0, 0.55, 0.5])
ax4 = plt.axes([0.67, 0, 0.35, 0.5])

ax1.plot(df_sorted['variables'], df_sorted['num_params'], marker='o', linestyle='-', 
         color='#384B70', linewidth=4, markersize=15, markeredgewidth=0, markerfacecolor='#384B70')
ax1.plot(df_sorted['variables'], df_sorted['num_params_2'], marker='o', linestyle='-', 
         color='#B8001F', linewidth=4, markersize=15, markeredgewidth=0, markerfacecolor='#B8001F')

ax1.set_xlabel('#variables')
ax1.set_ylabel('#segments')
ax1.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax1.set_yscale('function', functions=(lambda x: np.sqrt(x), lambda x: x**2))
ax1.set_yticks(np.arange(0, 2600, 500))
ax1.set_yticks(np.arange(0, 2600, 100), minor=True)
ax1.set_xticks(np.arange(0, 51, 10))
ax1.set_xticks(np.arange(0, 51, 5), minor=True)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

def plot_fit_line(x, y, color, ax, str):
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    ax.plot(x, poly(x), color=color, linestyle='--', linewidth=3)
    a, b, c = coef
    eq_str = f"{str}: $y = {a:.2f}x^2$"
    ax.plot([], [], linestyle='--', color=color, label=eq_str)

plot_fit_line(df_sorted['variables'], df_sorted['num_params'], '#384B70', ax1, "Theorem 1")
plot_fit_line(df_sorted['variables'], df_sorted['num_params_2'], '#B8001F', ax1, "pruned")
ax1.legend(
    loc='upper left', ncol=1, frameon=False,
    bbox_to_anchor=(0, 1), mode="expand", borderaxespad=0,
    fontsize=35
)


ratio_1 = df_sorted['depth'] / df_sorted['num_params']
ratio_2 = df_sorted['depth_2'] / df_sorted['num_params']

ax2.bar(df_sorted['variables'], ratio_1, width=2, color='#6A9C89', edgecolor='black', linewidth=2, label="transpiled circuit (quebec)")
ax2.bar(df_sorted['variables'], ratio_2, width=2, color='#afc4bc', hatch='/', edgecolor='black', linewidth=2, label="after Hamiltonian pruning")
ax2.legend(
    loc='upper left', ncol=1, frameon=False,
    bbox_to_anchor=(0, 1), mode="expand", borderaxespad=0,
    fontsize=50
)

ax2.set_xlabel('#variables')
ax2.set_ylabel('depth')
ax2.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax2.set_yticks(np.arange(0, 4000, 1000))
ax2.set_yticks(np.arange(0, 4000, 200), minor=True)
ax2.set_xticks(np.arange(0, 51, 10))
ax2.set_xticks(np.arange(0, 51, 5), minor=True)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# 

colors = ['#B8001F', '#FFF5E4', '#FFF']

arg_data = pivot_df_1['ARG']
bar_width = 0.35
pkid_values = arg_data.index.unique()
index = np.arange(len(pkid_values))

label_list = ["Choco-Q", "Rasengan"]
for idx, method in enumerate(arg_data.columns):
    ax3.bar(index + idx * bar_width, arg_data[method].values, bar_width,
            label=label_list[idx], color=colors[idx % len(colors)], edgecolor="black", linewidth=4)

ax3.legend(
    loc='upper left', ncol=1, frameon=False,
    bbox_to_anchor=(0, 1), mode="expand", borderaxespad=0,
    fontsize=50
)
ax3.set_xlabel('#variables', fontsize=60)
ax3.set_ylabel('ARG', fontsize=60)
ax3.set_xticks(index + bar_width * (len(arg_data.columns) / 2 - 0.5))
ax3.set_xticklabels(pkid_values)
ax3.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax3.set_yticks([i / 10 for i in range(0, 11, 5)])
ax3.set_yticks([i / 10 for i in range(0, 11, 1)], minor=True)

arg_data_2 = pivot_df_2['ARG']
bar_width = 0.4
pkid_values_2 = arg_data_2.index.unique()
index_2 = np.arange(len(pkid_values_2))

for idx, method in enumerate(arg_data_2.columns):
    ax4.bar(index_2 + idx * bar_width / 2, arg_data_2[method].values, bar_width,
            label=method, color=colors[1], edgecolor="black", linewidth=4)

ax4.set_xlabel('#variables', fontsize=60)
ax4.set_ylabel('ARG', fontsize=60)
ax4.set_xticks(index_2)
ax4.set_xticklabels(pkid_values_2)
ax4.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax4.set_ylim(0, 1)
ax4.set_yticks([i / 10 for i in range(0, 11, 5)])
ax4.set_yticks([i / 10 for i in range(0, 11, 1)], minor=True)

title = "Figure 10: Scalability analysis on large-scale FLP problems"
plt.suptitle(title, y=-0.2, fontsize=48)
plt.savefig(f'figure_10/{title}.svg', bbox_inches='tight')
print("Figuire 10 finished.")

num_cases = 10
flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2)], 1, 10)
problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

noisy_csv_path = "figure_11/fake_evaluate.csv"

solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSegmentedSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method', 'fake_device'] + evaluation_metrics
fake_devices = [FakeKyivProvider, FakeBrisbaneProvider]

def process_layer(prb, num_layers, solver, fake_device):
    opt = CobylaOptimizer(max_iter=50)
    fake_provider = fake_device()
    prb.set_penalty_lambda(400)
    solver_args = dict(
        prb_model=prb,
        optimizer=opt,
        provider=fake_provider,
        num_layers=num_layers,
        shots=1024,
    )
    if solver == RasenganSegmentedSolver:
        solver_args["num_segments"] = 1000
    used_solver = solver(**solver_args)
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

if __name__ == '__main__':
    print("Evaluating ARG:")
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 10 # Set timeout duration
    num_complete = 0
    # print(amp_csv_path)
    with open(f'{noisy_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    num_processes = num_processes_cpu // 4
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for fake_device_idx in range(len(fake_devices)):
            for solver_idx in range(len(solvers)):
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    layer = 5
                    for pbid, prb in enumerate(problems):
                        # Bug in the retrieval of Python class names
                        fake_device = fake_devices[fake_device_idx]
                        device_name = fake_device.__name__
                        solver = solvers[solver_idx]
                        solver_name = solver.__name__
                        future = executor.submit(process_layer, prb, layer, solver, fake_device)
                        futures.append((future, prb, pkid, pbid, layer, solver_name, device_name))

                
                    start_time = time.perf_counter()
                    for future, prb, pkid, pbid, layer, solver, fake_device in tqdm(futures, desc=f"{solver_name} on {device_name}"):
                        current_time = time.perf_counter()
                        remaining_time = max(set_timeout - (current_time - start_time), 0)
                        diff = []
                        try:
                            metrics = future.result(timeout=remaining_time)
                            diff.extend(metrics)
                            # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                            row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver, fake_device] + diff
                            with open(f'{noisy_csv_path}', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(row)  # Write row immediately
                            num_complete += 1
                            if num_complete == len(futures):
                                # print(f'problem_pkg_{pkid} has finished')
                                for process in executor._processes.values():
                                    os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {noisy_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 

fake_evaluate_csv_path = 'figure_11/fake_evaluate.csv'

df = pd.read_csv(fake_evaluate_csv_path)
df_avg = df.groupby(['method', 'fake_device'], as_index=False).agg({
    'ARG': ['mean', 'std'], 
    'in_constraints_probs': ['mean', 'std']
})



# Initialize the figure and axis settings
scale = 1
fig = plt.figure(figsize=(22 * scale, 12 * scale))

mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    "lines.markersize": 35,
    "lines.markeredgewidth": 4,
    "markers.fillstyle": 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

methods = ["HeaSolver", "PenaltySolver", "ChocoSolver", "RasenganSegmentedSolver"]
fake_devices = ['FakeKyivProvider', 'FakeBrisbaneProvider']
fake_device_names = ['Fake-Kyiv', 'Fake-Brisbane']
metrics = ['ARG', 'in_constraints_probs']

move = 0.2
visible_bar_width = 0.1
colors = ['#384B70', '#6A9C89', '#B8001F', '#FFF5E4']

for i, fake_device in enumerate(fake_devices):
    for j, metric in enumerate(metrics):
        ax = fig.add_axes((j * 0.7 + i * 0.23, 0, 0.23, 0.5))

        for idx, method in enumerate(methods):
            data = df_avg[df_avg['fake_device'] == fake_device]
            method_data = data[data['method'] == method]
            y = method_data[metric]['mean'].values.tolist()

            x_pos = idx * move

            if method == "RasenganSegmentedSolver":
                ax2 = ax.twinx()
                ax2.bar(
                    x_pos, y, visible_bar_width, color=colors[idx], 
                    edgecolor="black", label=method, linewidth=4,
                )

                if j == 0:
                    ax2.set_ylim(0, 2)
                else:
                    ax2.set_ylim(0, 105)

                if i == 1 and j == 0:
                    ax2.set_yticks([x / 10 for x in range(0, 10, 3)])
                else:
                    ax2.set_yticks([])

            else:
                ax.bar(
                    x_pos, y, visible_bar_width, color=colors[idx],
                    edgecolor="black", label=method, linewidth=4,
                )

        ax.set_xticks([])
        ax.set_xlim(-0.1, 0.7)
        ax.set_xlabel(fake_device_names[i])

        if j == 1:
            ax.set_ylim(0, 105)
        if i == 0:
            ax.set_ylabel('ARG' if j == 0 else 'in-constraints rate (%)')
        else:
            ax.set_yticks([])

# 图例
fig.legend(
    labels=["HEA", "P-QAOA", "Choco-Q", "Rasengan"],
    frameon=False,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.7),
    ncol=4
)

title = "Figure 11: Evaluation on real-world quantum platforms"
plt.suptitle(title, y=-0.13, fontsize=48)
plt.savefig(f'figure_11/{title}.svg', bbox_inches='tight')
print("Figuire 11 finished.")

device = detect_device()
print(f"Backend device: {device}")

num_cases = 2
problem_scale = 4

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)][:problem_scale], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)][:problem_scale], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)][:problem_scale])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)][:problem_scale])

latency_path = "figure_12/latency.csv"

problems_pkg = flp_problems_pkg + kpp_problems_pkg + jsp_problems_pkg + scp_problems_pkg + gcp_problems_pkg

metrics_lst = ['depth', 'num_params','latency_all']
solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSolver]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, solver, metrics_lst):
    opt = CobylaOptimizer(max_iter=300)
    provider = FakeQuebecProvider()
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = provider,
        num_layers = num_layers,
        shots = 1024*10,
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    
    with open(f'{latency_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
        futures = []
        id = 0
        for solver in solvers:
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    # print(f'process_{id} build')
                    id += 1
                    num_layers = 5
                    future = executor.submit(process_layer, problem, num_layers, solver, metrics_lst)
                    futures.append((future, pkid, solver.__name__, num_layers))

        start_time = time.perf_counter()
        for future, pkid, solver, num_layers in tqdm(futures, desc="Evaluating latency"):
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend(result)
                # print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
            finally:
                row = [pkid, solver, num_layers] + diff
                with open(f'{latency_path}', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    # print(f'Data has been written to {latency_path}')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {latency_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

evaluate_csv_path = 'figure_12/evaluate.csv'

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(kpp_problems_pkg),
        enumerate(jsp_problems_pkg),
        enumerate(scp_problems_pkg),
        enumerate(gcp_problems_pkg),
    )
)

solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider(transpile_mode=0)
    cpu = AerProvider()
    gpu = AerGpuProvider()
    baseline_provider = cpu if device == 'CPU' else gpu

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
    print("Evaluating ARG:")
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    # print(evaluate_csv_path)
    with open(f'{evaluate_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:
            # 防止GPU内存溢出
            if device == 'GPU' and solver in [HeaSolver, PenaltySolver]:
                num_processes = 2**(4 - diff_level)
            else:
                num_processes = num_processes_cpu // 4

            solver_name = solver.__name__
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

                for pbid, prb in enumerate(problems):
                    # print(f'{pkid}-{pbid}, {layer}, {solver} build')
                    future = executor.submit(process_layer, prb, layer, solver)
                    futures.append((future, prb, pkid, pbid, layer, solver_name))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver in tqdm(futures, desc=f"problem_{pkid} using {solver_name}"):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {evaluate_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

import pandas as pd

df = pd.read_csv(evaluate_csv_path)

df = df.drop(columns=['pbid'])
df = df[df['ARG'] <= 100000]
df['pkid_group'] = df['pkid'] % 4

df_quantum = pd.read_csv(latency_path)
df_quantum = df_quantum.groupby(['pkid', 'method'], as_index=False).agg({
    'latency_all': 'mean',
})
df_quantum['quantum'] = (df_quantum['latency_all'] * 300 * 1024) / 1e9

df = pd.merge(df, df_quantum[['pkid', 'method', 'quantum']], on=['pkid', 'method'], how='left', suffixes=('', '_new'))
df['quantum'] = df['quantum_new']
df.drop(columns=['quantum_new'], inplace=True)

mean_values = df.groupby('method').mean().reset_index()
mean_values['pkid_group'] = 5
df = pd.concat([df, mean_values], axis=0)

grouped_df = df.groupby(['pkid_group', 'method'], as_index=False).agg({
    'iteration_count': 'mean',
    'classcial': 'mean',
    'quantum': 'mean',
    'run_times': 'mean',
    "ARG": 'mean',
    'in_constraints_probs': 'mean',
    'best_solution_probs': 'mean',
})

values = ['classcial', 'quantum', 'ARG', 'best_solution_probs', 'run_times', 'iteration_count']
method_order = ['HeaSolver', 'PenaltySolver', 'ChocoSolver', 'RasenganSolver']
pivot_df = grouped_df.pivot(index='pkid_group', columns='method', values=values)
pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product([values, method_order]))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    "lines.markersize": 35,
    "lines.markeredgewidth": 4,
    "markers.fillstyle": 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

pie_scale = 0.25  # 控制饼图大小（相对于主图）
colors = ['#384B70', '#6A9C89', '#B8001F', '#FFF5E4']
colors_2 = ['#8d97a9', '#a9bfb7', '#c6767f', '#fffcf5']
hatches = ['/', '/', '/', '/']  # 对应四种方法的量子部分斜线

classcial_quantum_sums = {}
for method in method_order:
    data = grouped_df[(grouped_df.method == method) & (grouped_df.pkid_group != 5)]
    classcial_quantum_sums[method] = [data['classcial'].sum(), data['quantum'].sum()]

fig = plt.figure(figsize=(22, 12))
ax = fig.add_axes((0, 0, 1, 0.6))  # 主坐标轴
bar_width = 0.2

for idx, method in enumerate(method_order):
    data = grouped_df[grouped_df.method == method]
    x = np.arange(len(data.pkid_group))
    y_classcial = data['classcial'].to_list()
    y_quantum = data['quantum'].to_list()
    offset = idx * bar_width - (len(method_order) / 2 - 0.5) * bar_width

    ax.bar(x + offset, y_classcial, width=bar_width, color=colors[idx], edgecolor="black", linewidth=4)
    ax.bar(x + offset, y_quantum, width=bar_width, bottom=y_classcial, color=colors_2[idx], edgecolor="black",
           linewidth=4, hatch=hatches[idx])

ax.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax.set_ylabel('latency (s)')
ax.set_xticks(range(5))
ax.set_xticklabels([f"scale{i + 1}" for i in range(4)] + ["Avg."])

pie_positions = [
    (-0.05, 0.35, pie_scale, pie_scale),
    (0.08, 0.35, pie_scale, pie_scale),
    (0.21, 0.35, pie_scale, pie_scale),
    (-0.05, 0.13, pie_scale, pie_scale),
]
for i, (method, values) in enumerate(classcial_quantum_sums.items()):
    sub_ax = fig.add_axes(pie_positions[i])
    sub_ax.pie(values,
               colors=[colors[i], "black"],
               autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor': 'black', 'linewidth': 2},
               textprops={'fontsize': 24})

    wedges, _ = sub_ax.pie(values,
                            colors=["none", colors_2[i]],
                            startangle=90,
                            radius=1,
                            wedgeprops=dict(width=1, edgecolor='black', linewidth=2))
    wedges[1].set_hatch(hatches[i])

labels=["HEA", "P-QAOA", "Choco-Q", "Rasengan"]


handles = []
for i in range(4):
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="black", linewidth=2, label=f'c {labels[i]}'))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors_2[i], edgecolor="black", linewidth=2, hatch=hatches[i], label=f'q {labels[i]}'))


ax.legend(handles=handles, loc='upper left', ncol=4,
                    bbox_to_anchor=(- 0.2, 1.2, 1, 0.2), frameon=False)

title = "Figure 12: Latency breakdown of different methods"
plt.suptitle(title, y=-0.1, fontsize=48)
plt.savefig(f'figure_12/{title}.svg', bbox_inches='tight')
print("Figuire 12 finished.")

num_cases = 10

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(2, 3), (3, 3)], 10, 30)

latency_path = "figure_13/latency.csv"

problems_pkg = flp_problems_pkg

metrics_lst = ['depth', 'num_params','latency_all']
solvers = [RasenganSolver]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, solver, metrics_lst):
    opt = CobylaOptimizer(max_iter=300)
    provider = FakeQuebecProvider()
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = provider,
        num_layers = num_layers,
        shots = 1024*10,
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    
    with open(f'{latency_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
        futures = []
        id = 0
        for solver in solvers:
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    # print(f'process_{id} build')
                    id += 1
                    num_layers = 5
                    future = executor.submit(process_layer, problem, num_layers, solver, metrics_lst)
                    futures.append((future, pkid, solver.__name__, num_layers))

        start_time = time.perf_counter()
        for future, pkid, solver, num_layers in tqdm(futures, desc="Evaluating latency"):
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend(result)
                # print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
            finally:
                row = [pkid, solver, num_layers] + diff
                with open(f'{latency_path}', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    # print(f'Data has been written to {latency_path}')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {latency_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

segmented_csv_path = "figure_13/segmented_evaluate.csv"

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

solvers = [RasenganSegmentedSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method', 'num_segments'] + evaluation_metrics
num_segments_list = range(1, 11, 1)

def process_layer(prb, num_layers, solver, num_segments):
    opt = CobylaOptimizer(max_iter=300, tol=2e-1)
    ddsim = DdsimProvider()
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = ddsim,
        num_layers = num_layers,
        shots = 1024,
        num_segments = num_segments,
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
    # print(segmented_csv_path)
    with open(f'{segmented_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for num_segments in tqdm(num_segments_list, desc=f"Evaluating problem {pkid} across num_segments") :
            for solver in solvers:
                num_processes = 100

                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    layer = 5

                    for pbid, prb in enumerate(problems):
                        # print(f'{pkid}-{pbid}, {layer}, {solver} S={num_segments} build')
                        future = executor.submit(process_layer, prb, layer, solver, num_segments)
                        futures.append((future, prb, pkid, pbid, layer, solver.__name__, num_segments))

                    start_time = time.perf_counter()
                    for future, prb, pkid, pbid, layer, solver, num_segments, in futures:
                        current_time = time.perf_counter()
                        remaining_time = max(set_timeout - (current_time - start_time), 0)
                        diff = []
                        try:
                            metrics = future.result(timeout=remaining_time)
                            diff.extend(metrics)
                            # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} S={num_segments} executed successfully.")
                        except MemoryError:
                            print(f"Task for problem {pkid}-{pbid} L={layer} {solver} S={num_segments} encountered a MemoryError.")
                            for dict_term in evaluation_metrics:
                                diff.append('memory_error')
                        except TimeoutError:
                            print(f"Task for problem {pkid}-{pbid} L={layer} {solver} S={num_segments} timed out.")
                            for dict_term in evaluation_metrics:
                                diff.append('timeout')
                        except Exception as e:
                            print(f"An error occurred: {e}")
                        finally:
                            row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver, num_segments] + diff
                            with open(f'{segmented_csv_path}', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(row)  # Write row immediately
                            num_complete += 1
                            if num_complete == len(futures):
                                # print(f'problem_pkg_{pkid} has finished')
                                for process in executor._processes.values():
                                    os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {segmented_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

measure_latency = 1244.4444444444443

df = pd.read_csv(segmented_csv_path)
df = df.drop(columns=['pbid'])
df = df[df['ARG'] <= 100000]    

df_quantum = pd.read_csv(latency_path)
df_quantum = df_quantum.groupby(['pkid', 'method'], as_index=False).agg({
    'latency_all': 'mean',
})
df_quantum['quantum'] = (df_quantum['latency_all'] * 300 * 1024) / 1e9

df = pd.merge(df, df_quantum[['pkid', 'quantum']], on='pkid', how='left', suffixes=('', '_new'))
df['quantum'] = df['quantum_new']
df.drop(columns=['quantum_new'], inplace=True)

df['pkid'] = df['pkid'] % 2
df['quantum'] += measure_latency * (df['run_times'] - 300 * 1024) / 1e9

grouped_df = df.groupby(['pkid', 'num_segments', 'method'], as_index=False).agg({
    'iteration_count': 'mean',
    'classcial': 'max',
    'quantum': 'mean',
    'run_times': 'mean',
})



# Set global plot style
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    "lines.markersize": 25,
    "lines.markeredgewidth": 4,
    "markers.fillstyle": 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

# Initialize figure and axes
scale = 1
fig = plt.figure(figsize=(22 * scale, 12 * scale))
ax1 = plt.axes((0, 0, 0.4, 0.5))
ax2 = plt.axes((0.5, 0, 0.4, 0.5))

# Colors and line styles
colors_ = ['#384B70', '#6A9C89']         # For run_times
colors_2 = ['#8d97a9', '#a9bfb7']         # For run_times
colors_quantum = ['#B8001F', '#FF7F00']  # For quantum (not used here)
line_styles = ['-', '--']

label_list = ["F2", "F3"]

# Plot run_times line chart on ax1
for idx, pkid_value in enumerate([0, 1]):
    data = grouped_df[grouped_df['pkid'] == pkid_value]
    x = data['num_segments']
    marker_style = 's' if pkid_value == 0 else 'o'
    
    ax1.plot(
        x, data['run_times'],
        color=colors_[idx],
        linestyle=line_styles[idx],
        label=label_list[idx],
        linewidth=4,
        marker=marker_style,
        markerfacecolor=colors_[idx],
        markeredgecolor='black'
    )

ax1.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax1.set_xlabel('#segments')
ax1.set_ylabel('#shots')
ax1.legend(
    loc='upper left', ncol=2, frameon=False,
    bbox_to_anchor=(0, 1.1, 1, 0.3), mode="expand", borderaxespad=0
)

# Plot stacked bar chart of classical + quantum on ax2
bar_width = 0.35

class_label_list = ["c F2", "c F3"]
quantum_label_list = ["q F2", "q F3"]

for idx, pkid_value in enumerate([0, 1]):
    data = grouped_df[grouped_df['pkid'] == pkid_value]
    x = np.arange(len(data['num_segments']))
    y_classical = data['classcial'].to_list()
    y_quantum = data['quantum'].to_list()

    ax2.bar(
        x + idx * bar_width, y_classical,
        width=bar_width, color=colors_[idx],
        label=class_label_list[idx],
        edgecolor="black", linewidth=4
    )

    ax2.bar(
        x + idx * bar_width, y_quantum,
        width=bar_width, bottom=y_classical,
        color=colors_2[idx], hatch='/',
        label=quantum_label_list[idx],
        edgecolor="black", linewidth=4
    )

ax2.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax2.set_xlabel('#segments')
ax2.set_ylabel('latency (s)')
ax2.legend(
    loc='upper left', ncol=2, frameon=False,
    bbox_to_anchor=(0, 1.1, 1, 0.4), mode="expand", borderaxespad=0,
    fontsize='small'
)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Save and display
# plt.tight_layout()
title = 'Figure 13: Shots and latency of Rasengan with different numbers of segments'
plt.suptitle(title, y=-0.2, fontsize=48)
plt.savefig(f'figure_13/{title}.svg', bbox_inches='tight')
print("Figuire 13 finished.")

depolarizing_csv_path = "figure_14/depolarizing.csv"
num_cases_1 = 30
flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases_1, [(1, 2)], 1, 20)
problems_pkg_1 = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

amp_csv_path = "figure_14/amp_damping_probability.csv"
num_cases_2 = 10
flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases_2, [(1, 2)], 1, 10)
problems_pkg_2 = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method', 'p_gate1'] + evaluation_metrics

solvers = [RasenganSegmentedSolver]
p_gate1_lst = [1e-4,3e-4,5e-4,1e-3]

def process_layer(prb, num_layers, solver,p):
    opt = CobylaOptimizer(max_iter=200)
    aer = BitFlipNoiseAerProvider(p_meas= 1.525e-2, p_reset=p, p_gate1=p)
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = aer,
        num_layers = num_layers,
        shots = 1024,
        num_segments = 100,

    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    result = eval + time + [run_times]
    return result


if __name__ == '__main__':
    
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    with open(f'{depolarizing_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    for pkid, (diff_level, problems) in enumerate(problems_pkg_1):
        for p in tqdm(p_gate1_lst, desc="Evaluating p_gate1"):
            num_processes = num_processes_cpu // 4

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5
                solver = RasenganSegmentedSolver
                for pbid, prb in enumerate(problems):
                    future = executor.submit(process_layer, prb, layer, solver, p)
                    futures.append((future, prb, pkid, pbid, layer, solver.__name__,p))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver, p in tqdm(futures, desc="    processing", leave=False):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} p_gate1={p} has executed successfully.")
                    except MemoryError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} p_gate1={p} encountered a MemoryError.")
                        for dict_term in evaluation_metrics:
                            diff.append('memory_error')
                    except TimeoutError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} p_gate1={p} timed out.")
                        for dict_term in evaluation_metrics:
                            diff.append('timeout')
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver, p] + diff
                        with open(f'{depolarizing_csv_path}', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {depolarizing_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

solvers = [RasenganSegmentedSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method', 'amp'] + evaluation_metrics


def process_layer(prb, num_layers, solver, amp):
    opt = CobylaOptimizer(max_iter=200)
    noisy_ddsim = NoisyDdsimProvider(amp_damping_probability=amp)
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = noisy_ddsim,
        num_layers = num_layers,
        shots = 1024,
        num_segments=1000,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]


amp_damping_probability = [0.0, 0.005, 0.01, 0.015]
    

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 10 # Set timeout duration
    num_complete = 0
    # print(amp_csv_path)
    with open(f'{amp_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg_2):
        # if pkid % 5 == 0:
        #     continue
        for amp in tqdm(amp_damping_probability, desc="Evaluating amp_damping_probability"):
            for solver in solvers:
                num_processes = num_processes_cpu // 4

                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    layer = 5
                
                    for pbid, prb in enumerate(problems):

                        # print(f'{pkid}-{pbid}, {layer}, {solver} build')
                        future = executor.submit(process_layer, prb, layer, solver, amp)
                        futures.append((future, prb, pkid, pbid, layer, solver.__name__, amp))

                    start_time = time.perf_counter()
                    for future, prb, pkid, pbid, layer, solver, amp in tqdm(futures, desc="    processing", leave=False):
                        current_time = time.perf_counter()
                        remaining_time = max(set_timeout - (current_time - start_time), 0)
                        diff = []
                        try:
                            metrics = future.result(timeout=remaining_time)
                            diff.extend(metrics)
                            # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                            row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver, amp] + diff
                            with open(f'{amp_csv_path}', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(row)  # Write row immediately
                            num_complete += 1
                            if num_complete == len(futures):
                                # print(f'problem_pkg_{pkid} has finished')
                                for process in executor._processes.values():
                                    os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {amp_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

depolarizing_csv_path = "figure_14/depolarizing.csv"
df = pd.read_csv(depolarizing_csv_path)
df = df[df["ARG"] < 0.15]
cols_to_avg = [col for col in df.columns[6:] if col != 'p_gate1']
df[cols_to_avg] = df[cols_to_avg].apply(pd.to_numeric, errors='coerce')

df_hard = pd.read_csv(amp_csv_path)
df_hard = df_hard.drop(columns=['pbid'])
grouped_df = df_hard.groupby(['amp'], as_index=False).agg({
    'ARG': ['mean', 'std'],
})



scale = 1
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60 * scale,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5 * scale,
    'xtick.major.size': 20 * scale,
    'xtick.major.width': 5 * scale,
    'xtick.minor.size': 10 * scale,
    'xtick.minor.width': 3 * scale,
    'ytick.major.size': 20 * scale,
    'ytick.major.width': 5 * scale,
    'ytick.minor.size': 10 * scale,
    'ytick.minor.width': 3 * scale,
    'lines.markersize': 35 * scale,
    'lines.markeredgewidth': 4 * scale,
    'markers.fillstyle': 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
    'hatch.linewidth': 0.2 * scale,
    'hatch.color': 'black',
    'lines.linewidth': 0.7 * scale,
})

fig = plt.figure(figsize=(22 * scale, 12 * scale))
ax1 = plt.axes((0, 0, 0.415, 0.5))
ax2 = plt.axes((0.55, 0, 0.4, 0.5))

bar_width = 0.15
colors = ['#FFF5E4','#B8001F', '#6A9C89','#384B70']
x = np.linspace(0, 1, 1000)
for i, category in enumerate([1e-4, 3e-4, 5e-4, 1e-3]):
    subset = df[df["p_gate1"] == category]["ARG"]
    subset = pd.to_numeric(subset, errors='coerce').dropna()
    subset = subset[np.isfinite(subset)]
    if len(subset) == 0:
        continue
    mu, sigma = norm.fit(subset)
    pdf = norm.pdf(x, mu, sigma)
    ax1.plot(x, pdf, color='black', linewidth=3, zorder=1)
    ax1.fill_between(x, pdf, color=colors[i], label=f'{category:.0e} (μ={mu:.3f}, σ={sigma:.3f})', zorder=0)


ax1.legend(loc='upper right', ncol=1, frameon=False, prop={'size': 33})
ax1.set_xlabel('ARG')
ax1.set_ylabel('Probability Density')
ax1.set_xlim(0, 0.15)
ax1.set_ylim(0, 38)

bar_width = 0.5
colors = ['#FFF5E4', '#B8001F', '#6A9C89', '#384B70']
target_amps = [0.0, 0.005, 0.01, 0.015, 0.02]
x = np.arange(len(target_amps))

for i, amp in enumerate(target_amps):
    # 找 grouped_df 里是否有对应的 amp 行
    row = grouped_df[grouped_df['amp'] == amp]
    if not row.empty:
        y_val = row[('ARG', 'mean')].values[0]
        color = colors[0]
        ax2.bar(
            x[i], y_val, bar_width,
            color=color, edgecolor="black", linewidth=4 * scale, label=f'{amp:.3f}'
        )
    else:
        ax2.bar(
            x[i], 2, bar_width,
            color='white', edgecolor="black", linewidth=4 * scale, hatch='/',
            label=f'{amp:.3f} (missing)'
        )

ax2.set_xlabel('amplitude damping probability (%)', fontsize=45)
ax2.set_ylabel('ARG')
ax2.set_xticks(x, labels=[x_label * 100 for x_label in target_amps])
ax2.set_ylim(0, 1.5)
ax2.grid(True, linestyle='--', linewidth=1.5 * scale, axis='y')


title = "Figure 14: Evaluation on different noise models"
plt.suptitle(title, y=-0.18, fontsize=48)
plt.savefig(f'figure_14/{title}.svg', bbox_inches='tight')
print("Figuire 14 finished.")

num_cases = 10

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2)], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3)], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3)], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4)])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1)])

ablation_depth_csv_path = "figure_15/ablation_depth.csv"

problems_pkg = flp_problems_pkg + kpp_problems_pkg + jsp_problems_pkg + scp_problems_pkg + gcp_problems_pkg

metrics_lst = ['depth', 'num_params']
opt_modes = [[0, 0], [1, 0], [1, 1]]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, opt_mode, metrics_lst):
    opt = CobylaOptimizer(max_iter=300)
    ddsim = DdsimProvider()
    used_solver = RasenganSolver(
        prb_model = prb,
        optimizer = opt,
        provider = ddsim,
        num_layers = num_layers,
        shots = 1024,
        opt_mode = opt_modes[opt_mode],
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    
    with open(f'{ablation_depth_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
        futures = []
        id = 0
        for opt_mode in range(3):
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    # print(f'process_{id} build')
                    id += 1
                    num_layers = 5
                    future = executor.submit(process_layer, problem, num_layers, opt_mode, metrics_lst)
                    futures.append((future, pkid, opt_mode, num_layers))

        start_time = time.perf_counter()
        for future, pkid, opt_mode, num_layers in tqdm(futures, desc="Evaluating depth across opt_mode"):
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend(result)
                # print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
            finally:
                row = [pkid, opt_mode, num_layers] + diff
                with open(f'{ablation_depth_csv_path}', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    # print(f'Data has been written to {ablation_depth_csv_path}')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {ablation_depth_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")


ablation_depth_csv_path = "figure_15/ablation_depth.csv"
df = pd.read_csv(ablation_depth_csv_path)

df_to_modify = df[df['method'] == 2].copy()
df_to_modify['depth'] = df_to_modify['depth'] / df_to_modify['num_params']
df_to_modify['method'] = 3
df = pd.concat([df, df_to_modify], ignore_index=True)
df = df[df['depth'] >= 20]
df = df[(df['num_params'] >= 1)]

grouped_df = df.groupby(['pkid', 'method'], as_index=False).agg({
    "depth": ["mean", "std"],
    "num_params": ["mean", "std"],
})

values = ["depth", 'num_params']



mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    'lines.markersize': 35,
    'lines.markeredgewidth': 4,
    'markers.fillstyle': 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

scale = 1
fig = plt.figure(figsize=(22 * scale, 12 * scale))
ax = plt.axes((0, 0, 1, 0.6))

bar_width = 0.2
colors = ['#384B70', '#6A9C89', '#B8001F', '#FFF5E4']
labels = ['w/o opt', 'opt 1', 'opt 1+2', 'opt 1+2+3']
error_params = dict(elinewidth=3, ecolor='black', capsize=8, capthick=2)

for idx in range(4):
    data = grouped_df[grouped_df.method == idx]
    x = np.arange(len(data.pkid) + 1)
    y = data['depth']['mean'].to_list()
    y.append(np.mean(y))
    err = data['depth']['std'].to_list()
    err.append(0)

    ax.bar(
        x + idx * bar_width - bar_width,
        y,
        bar_width,
        color=colors[idx],
        yerr=err,
        error_kw=error_params,
        edgecolor="black",
        label=labels[idx],
        linewidth=4,
    )

ax.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(['F1', 'K1', 'J1', 'S1', 'G1', 'Avg.'])
ax.set_xlabel('Benchmark')
ax.set_ylabel('Depth')
ax.set_yticks([0, 500, 1000, 1500, 2000])

ax.legend(
    loc='upper left',
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0, 1.0, 1, 0.2),
    mode="expand",
    borderaxespad=0,
)

title = "Figure 15: Ablation study of optimization strategies on circuit depth"
plt.suptitle(title, y=-0.18, fontsize=48)
plt.savefig(f'figure_15/{title}.svg', bbox_inches='tight')
print("Figuire 15 finished.")

num_cases = 5

gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1)])

algorithmic_evaluate_csv_path = 'figure_16/algorithmic_evaluate.csv'

problems_pkg = list(
    itertools.chain(
        enumerate(gcp_problems_pkg),
    )
)

opt_modes = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, opt_mode):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider(transpile_mode=0)

    prb.set_penalty_lambda(400)
    used_solver = RasenganSegmentedSolver(
        prb_model = prb,
        optimizer = opt,
        provider = ddsim,
        num_layers = num_layers,
        shots = 2048,
        opt_mode = opt_mode,
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
    # print(algorithmic_evaluate_csv_path)
    with open(f'{algorithmic_evaluate_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for opt_mode in tqdm(opt_modes, desc="Algorithmic evaluation across opt_mode"):
            num_processes = num_processes_cpu // 4

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

                for pbid, prb in enumerate(problems):
                    # print(f'{pkid}-{pbid}, {layer}, {opt_mode} build')
                    future = executor.submit(process_layer, prb, layer, opt_mode)
                    futures.append((future, prb, pkid, pbid, layer, opt_mode))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, opt_mode in tqdm(futures, desc="    processing", leave=False):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {opt_mode} executed successfully.")
                    except MemoryError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {opt_mode} encountered a MemoryError.")
                        for dict_term in evaluation_metrics:
                            diff.append('memory_error')
                    except TimeoutError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {opt_mode} timed out.")
                        for dict_term in evaluation_metrics:
                            diff.append('timeout')
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), opt_mode] + diff
                        with open(f'{algorithmic_evaluate_csv_path}', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {algorithmic_evaluate_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

noisy_evaluate_csv_path = 'figure_16/noisy_evaluate.csv'

problems_pkg = list(
    itertools.chain(
        enumerate(gcp_problems_pkg),
    )
)

opt_modes = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, opt_mode):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = NoisyDdsimProvider(transpile_mode = 0)

    prb.set_penalty_lambda(400)
    used_solver = RasenganSegmentedSolver(
        prb_model = prb,
        optimizer = opt,
        provider = ddsim,
        num_layers = num_layers,
        shots = 1024,
        opt_mode = opt_mode,
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
    # print(noisy_evaluate_csv_path)
    with open(f'{noisy_evaluate_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for opt_mode in tqdm(opt_modes, desc="Noisy evaluation across opt_mode"):
            num_processes = num_processes_cpu // 4

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

                for pbid, prb in enumerate(problems):
                    # print(f'{pkid}-{pbid}, {layer}, {opt_mode} build')
                    future = executor.submit(process_layer, prb, layer, opt_mode)
                    futures.append((future, prb, pkid, pbid, layer, opt_mode))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, opt_mode in tqdm(futures, desc="    processing", leave=False):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {opt_mode} executed successfully.")
                    except MemoryError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {opt_mode} encountered a MemoryError.")
                        for dict_term in evaluation_metrics:
                            diff.append('memory_error')
                    except TimeoutError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {opt_mode} timed out.")
                        for dict_term in evaluation_metrics:
                            diff.append('timeout')
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), opt_mode] + diff
                        with open(f'{noisy_evaluate_csv_path}', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {noisy_evaluate_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")


algorithmic_df = pd.read_csv(algorithmic_evaluate_csv_path)
algorithmic_grouped_df = algorithmic_df.groupby(['method'], as_index=False).mean()

noisy_df = pd.read_csv(noisy_evaluate_csv_path)
noisy_grouped_df = noisy_df.groupby(['method'], as_index=False).mean()


# 图形风格设置
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    'lines.markersize': 35,
    'lines.markeredgewidth': 4,
    'markers.fillstyle': 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

# 设置方法和对应标签
methods = ['[0, 0, 0, 0]', '[1, 0, 0, 0]', '[1, 1, 0, 0]', '[1, 1, 1, 1]']
x_labels = ['w/o', '1', '1+2', '1+2+3']
x = np.arange(len(methods))
bar_width = 0.4

# 颜色
color_algo = '#FFF5E4'
color_classical = '#384B70'
edge_color = 'black'

# 错误棒参数
error_params = dict(elinewidth=3, ecolor='black', capsize=8, capthick=2)

# 图形创建
scale = 1
fig = plt.figure(figsize=(22 * scale, 12 * scale))
ax1 = plt.axes((0, 0, 0.35, 0.5))
ax1_1 = ax1.twinx()
ax2 = plt.axes((0.55, 0, 0.35, 0.5))

# 左图：ARG
for i, method in enumerate(methods):
    data_algo = algorithmic_grouped_df[algorithmic_grouped_df.method == method]
    data_noisy = noisy_grouped_df[noisy_grouped_df.method == method]

    ax1.bar(
        x[i],
        data_noisy['ARG'].values[0],
        width=bar_width,
        color=color_classical,
        edgecolor=edge_color,
        label='noisy' if i == 0 else "",
        error_kw=error_params,
        linewidth=4
    )
    
    ax1_1.bar(
        x[i],
        data_algo['ARG'].values[0],
        width=bar_width,
        color=color_algo,
        edgecolor=edge_color,
        label='algorithmic' if i == 0 else "",
        error_kw=error_params,
        linewidth=4
    )


ax1.set_ylabel('ARG')
ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=30)
ax1.set_xlim(-0.6, len(x) - 0.4)
ax1.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax1.set_ylim(0.01, 500)
ax1_1.set_ylim(0, 0.15)

# 右图：In-Constraints Rate
for i, method in enumerate(methods):
    data_algo = algorithmic_grouped_df[algorithmic_grouped_df.method == method]
    data_noisy = noisy_grouped_df[noisy_grouped_df.method == method]

    ax2.bar(
        x[i],
        data_algo['in_constraints_probs'].values[0],
        width=bar_width,
        color=color_algo,
        edgecolor=edge_color,
        label='algorithmic' if i == 0 else "",
        error_kw=error_params,
        linewidth=4
    )

    ax2.bar(
        x[i],
        data_noisy['in_constraints_probs'].values[0],
        width=bar_width,
        color=color_classical,
        edgecolor=edge_color,
        label='noisy' if i == 0 else "",
        error_kw=error_params,
        linewidth=4
    )

ax2.set_ylabel('in-constraints rate (%)')
# ax2.set_ylim(0, 105)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=30)
ax2.set_xlim(-0.6, len(x) - 0.4)
ax2.grid(True, linestyle='--', linewidth=1.5, axis='y')
ax2.set_yscale('log')

# 图例：只画一次
handles, labels = ax2.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(-0.1, 0.55, 1, 0.2),

)

title = "Figure 16: Ablation study on ARG and in-constraints rate"
plt.suptitle(title, y=-0.18, fontsize=48)
plt.savefig(f'figure_16/{title}.svg', bbox_inches='tight')
print("Figuire 16 finished.")

num_cases = 1

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)])

problems_pkg = [flp_problems_pkg, kpp_problems_pkg, scp_problems_pkg, gcp_problems_pkg]

space_list = []

for problem in tqdm(problems_pkg, desc="Analysing space"):
    problem_space_list = []
    for benchmark in tqdm(problem, desc="    processing", leave=False):
        opt = CobylaOptimizer(max_iter=200)
        aer = DdsimProvider()
        benchmark[0].set_penalty_lambda(200)
        solver = QtoExplorer(
            prb_model=benchmark[0],
            optimizer=opt,
            provider=aer,
            shots=1000,
        )
        exp_list, _, _ = solver.explore()
        problem_space_list.append(exp_list)
    space_list.append(problem_space_list)

with open("figure_17/space_explore.json", "w") as f:
    json.dump(space_list, f)

# read
with open("figure_17/space_explore.json", "r") as f:
    space_list = json.load(f)

# 全局绘图参数设置
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'Times New Roman',
    'font.size': 60,
    'axes.unicode_minus': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 5,
    'xtick.major.size': 20,
    'xtick.major.width': 5,
    'xtick.minor.size': 10,
    'xtick.minor.width': 3,
    'ytick.major.size': 20,
    'ytick.major.width': 5,
    'ytick.minor.size': 10,
    'ytick.minor.width': 3,
    "lines.markersize": 35,
    "lines.markeredgewidth": 4,
    "markers.fillstyle": 'full',
    'lines.markerfacecolor': '#f8d941',
    'lines.markeredgecolor': 'black',
})

# 图形参数与初始化
plt_x, plt_y = 0.3, 0.45
box_x, box_y = 0.4, 0.65
fig = plt.figure(figsize=(22, 12))
axes = [
    plt.axes((0, box_y, plt_x, plt_y)),
    plt.axes((0, 0, plt_x, plt_y)),
    plt.axes((box_x, box_y, plt_x, plt_y)),
    plt.axes((box_x, 0, plt_x, plt_y)),
]
bar_ax = plt.axes((0.85, 0, 0.35, 1.1))

colors = ['#FFF5E4', '#384B70', '#B8001F', '#6A9C89']
colors_2 = ['#fffcf5', '#8d97a9', '#c6767f', '#a9bfb7']
labels = ['HeaSolver', 'PenaltySolver', 'ChocoSolver', 'QtoSimplifyDiscardSegmentedSolver']
cal_list = [0] * 4
cal_list_2 = [0] * 4

# 均匀抽样函数
def get_uniform_points(x, y, max_points=30):
    indices = np.linspace(0, len(x) - 1, num=max_points, dtype=int)
    return [x[i] for i in indices], [y[i] for i in indices]

# 遍历每组图和数据
for ax, data, idx in zip(axes, space_list, range(4)):
    processed_data = []
    for d in data:
        unique = sorted(set(d))
        filled = unique + [unique[-1]] * (len(d) - len(unique))
        processed_data.append(filled)

    # 计算统计
    max_orig = [d.index(max(d)) for d in data]
    max_proc = [d.index(max(d)) for d in processed_data]
    lengths = [len(d) for d in data]

    for i, (a, b, L) in enumerate(zip(max_orig, max_proc, lengths)):
        cal_list[i] += a / L / 4 * 100
        cal_list_2[i] += b / L / 4 * 100

    # 绘图
    for i, (orig, proc) in enumerate(zip(data, processed_data)):
        x_orig = np.linspace(0, 100, len(orig))
        x_proc = np.linspace(0, 100, len(proc))
        ax.plot(x_orig, orig, label=f'{labels[i]} (Original)', color=colors[i], linestyle='--', linewidth=7)
        ax.plot(x_proc, proc, label=f'{labels[i]} (Processed)', color=colors[i], linestyle='-', linewidth=7)

# 绘制条形图
for i in range(4):
    bar_ax.barh(i, cal_list[i], height=0.25, color=colors_2[i], edgecolor="black", linewidth=4)
    bar_ax.barh(i, cal_list_2[i], height=0.25, color=colors[i], edgecolor="black", linewidth=4, alpha=1)

bar_ax.set_ylim(-0.5, 3.5)
bar_ax.set_yticks([])

title = "Figure 17: Solution space analysis on Hamiltonian Pruning"
plt.suptitle(title, y=-0.1, fontsize=48)
plt.savefig(f'figure_17/{title}.svg', bbox_inches='tight')
print("Figuire 17 finished.")

device = detect_device()
print(f"Backend device: {device}")

num_cases = 8
problem_scale = 4

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)][:problem_scale], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)][:problem_scale], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)][:problem_scale])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)][:problem_scale])

evaluate_csv_path = 'table_2/evaluate.csv'

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(kpp_problems_pkg),
        enumerate(jsp_problems_pkg),
        enumerate(scp_problems_pkg),
        enumerate(gcp_problems_pkg),
    )
)

solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider(transpile_mode=0)
    cpu = AerProvider()
    gpu = AerGpuProvider()
    baseline_provider = cpu if device == 'CPU' else gpu

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
    print("Evaluating ARG:")
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    # print(evaluate_csv_path)
    with open(f'{evaluate_csv_path}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:
            # 防止GPU内存溢出
            if device == 'GPU' and solver in [HeaSolver, PenaltySolver]:
                num_processes = 2**(4 - diff_level)
            else:
                num_processes = num_processes_cpu // 4

            solver_name = solver.__name__
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5

                for pbid, prb in enumerate(problems):
                    # print(f'{pkid}-{pbid}, {layer}, {solver} build')
                    future = executor.submit(process_layer, prb, layer, solver)
                    futures.append((future, prb, pkid, pbid, layer, solver_name))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver in tqdm(futures, desc=f"problem_{pkid} using {solver_name}"):
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        # print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
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
                            # print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {evaluate_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")

depth_and_num_params_csv_path = 'table_2/depth_and_num_params.csv'
problems_pkg = flp_problems_pkg + kpp_problems_pkg + jsp_problems_pkg + scp_problems_pkg + gcp_problems_pkg
metrics_lst = ['depth', 'num_params']
solvers = [HeaSolver, PenaltySolver, ChocoSolver, RasenganSolver]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, solver, metrics_lst):
    opt = CobylaOptimizer(max_iter=300)
    ddsim = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    baseline_provider = cpu if device == 'CPU' else gpu

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
    all_start_time = time.perf_counter()
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
        for future, pkid, solver, num_layers in tqdm(futures, desc="Evaluating depth and num_params"):
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend(result)
                # print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
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
                    # print(f'Data has been written to {depth_and_num_params_csv_path}')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {depth_and_num_params_csv_path}')
    print(f"Time elapsed: {time.perf_counter() - all_start_time:.2f}s")


depth_and_num_params_csv_path = 'table_2/depth_and_num_params.csv'
evaluate_csv_path = 'table_2/evaluate.csv'
problem_scale = 4

# 设置显示选项
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 输入路径
depth_and_num_params_csv_path = 'table_2/depth_and_num_params.csv'
evaluate_csv_path = 'table_2/evaluate.csv'
problem_scale = 4

# 读入 Depth 和 Params 数据
df = pd.read_csv(depth_and_num_params_csv_path)
df.loc[df["method"] == "HeaSolver", "method"] = "HEA"
df.loc[df["method"] == "PenaltySolver", "method"] = "P-QAOA"
df.loc[df["method"] == "ChocoSolver", "method"] = "Choco-Q"
df.loc[df["method"] == "RasenganSolver", "method"] = "Rasengan"

benchmarks = ["F", "K", "J", "S", "G"]
method_order = ['HEA', 'P-QAOA', 'Choco-Q', 'Rasengan']

# pkid → Benchmark label，如 F1, F2, ..., G4
pkid_to_label = {
    pkid: f"{b}{i}"
    for pkid, (b, i) in enumerate(
        ((b, i) for b in benchmarks for i in range(1, problem_scale + 1))
    )
}
df["Benchmark"] = df["pkid"].map(pkid_to_label)
df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

# 聚合
df_grouped = df.groupby(["method", "Benchmark"], observed=True).mean(numeric_only=True).reset_index()

# 构建 pivot 表格
pivot_depth = df_grouped.pivot(index="method", columns="Benchmark", values="depth")
pivot_param = df_grouped.pivot(index="method", columns="Benchmark", values="num_params")
pivot_depth.loc["Rasengan"] = pivot_depth.loc["Rasengan"] / pivot_param.loc["Rasengan"]

# ARG 数据处理
df_eval = pd.read_csv(evaluate_csv_path)
df_eval.loc[df_eval["method"] == "HeaSolver", "method"] = "HEA"
df_eval.loc[df_eval["method"] == "PenaltySolver", "method"] = "P-QAOA"
df_eval.loc[df_eval["method"] == "ChocoSolver", "method"] = "Choco-Q"
df_eval.loc[df_eval["method"] == "RasenganSolver", "method"] = "Rasengan"
df_eval = df_eval[df_eval['ARG'] <= 100000]
df_eval["Benchmark"] = df_eval["pkid"].map(pkid_to_label)
df_eval["method"] = pd.Categorical(df_eval["method"], categories=method_order, ordered=True)
df_arg_grouped = df_eval.groupby(["method", "Benchmark"], observed=True)["ARG"].mean().reset_index()
pivot_arg = df_arg_grouped.pivot(index="method", columns="Benchmark", values="ARG")

# 列顺序对齐
column_order = list(pkid_to_label.values())
pivot_depth = pivot_depth[column_order]
pivot_param = pivot_param[column_order]
pivot_arg = pivot_arg[column_order]

# 四舍五入

# 多级索引
depth_labeled = pivot_depth.copy()
depth_labeled.index = pd.MultiIndex.from_product([["Depth"], depth_labeled.index])
param_labeled = pivot_param.copy()
param_labeled.index = pd.MultiIndex.from_product([["#Params"], param_labeled.index])
arg_labeled = pivot_arg.copy()
arg_labeled.index = pd.MultiIndex.from_product([["ARG"], arg_labeled.index])

# === 计算 improvement ===
def compute_improvement(target_df, baseline="Rasengan"):
    improvements = {}
    for method in method_order:
        if method == baseline:
            improvements[method] = None
            continue
        ratio = target_df.loc[method] / target_df.loc[baseline]
        improvements[method] = ratio.mean()
    return improvements

def append_improvement_column(df, improvements):
    df = df.copy()
    df["improvement"] = [
        round(val, 3) if val is not None else pd.NA
        for val in [improvements.get(method, pd.NA) for method in df.index.get_level_values(1)]
    ]
    return df

# 分别计算 improvement 并添加列
arg_improvements = compute_improvement(pivot_arg)
depth_improvements = compute_improvement(pivot_depth)
param_improvements = compute_improvement(pivot_param)

depth_labeled = depth_labeled.round().astype("Int64").astype(str)
param_labeled = param_labeled.round().astype("Int64").astype(str)
arg_labeled = arg_labeled.round(3)

arg_labeled = append_improvement_column(arg_labeled, arg_improvements)
depth_labeled = append_improvement_column(depth_labeled, depth_improvements)
param_labeled = append_improvement_column(param_labeled, param_improvements)

# 合并最终表格
merged_with_improvement = pd.concat([arg_labeled, depth_labeled, param_labeled])
merged_with_improvement.to_pickle("table_2/table_2.pkl")

print("Table 2 finished.")