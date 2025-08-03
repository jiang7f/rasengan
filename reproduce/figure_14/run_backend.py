import os
import time
import csv
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
import numpy as np
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import (
    RasenganSegmentedSolver, BitFlipNoiseAerProvider, NoisyDdsimProvider
)
np.random.seed(0x7f)
random.seed(0x7f)

depolarizing_csv_path = "depolarizing.csv"
num_cases_1 = 30
flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases_1, [(1, 2)], 1, 20)
problems_pkg_1 = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
    )
)

amp_csv_path = "amp_damping_probability.csv"
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

depolarizing_csv_path = "depolarizing.csv"
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
plt.savefig(f'{title}.svg', bbox_inches='tight')
print("Figuire 14 finished.")