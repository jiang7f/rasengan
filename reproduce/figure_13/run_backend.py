import os
import time
import csv
import signal
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from rasengan.problems.facility_location_problem import generate_flp
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import (
    RasenganSolver, RasenganSegmentedSolver, DdsimProvider, FakeQuebecProvider
)

num_cases = 10

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(2, 3), (3, 3)], 10, 30)

latency_path = "latency.csv"

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

segmented_csv_path = "segmented_evaluate.csv"

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
plt.savefig(f'{title}.svg', bbox_inches='tight')
print("Figuire 13 finished.")