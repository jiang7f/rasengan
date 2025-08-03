import os
import time
import csv
import random
import signal
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from rasengan.problems.facility_location_problem import generate_flp
from rasengan.problems.k_partition_problem import generate_kpp
from rasengan.problems.job_scheduling_problem import generate_jsp
from rasengan.problems.set_cover_problem import generate_scp
from rasengan.problems.graph_coloring_problem import generate_gcp
from rasengan.solvers.optimizers import CobylaOptimizer
from rasengan.solvers.qiskit import (
    HeaSolver, PenaltySolver, ChocoSolver, RasenganSolver,
    AerGpuProvider, AerProvider, DdsimProvider,
)
from rasengan.utils.qiskit_plugin import detect_device

device = detect_device()
print(f"Backend device: {device}")

num_cases = 8
problem_scale = 4

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 10, 30)
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)][:problem_scale], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)][:problem_scale], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)][:problem_scale])
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)][:problem_scale])

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

depth_and_num_params_csv_path = 'depth_and_num_params.csv'
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


depth_and_num_params_csv_path = 'depth_and_num_params.csv'
evaluate_csv_path = 'evaluate.csv'
problem_scale = 4

# 设置显示选项
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 输入路径
depth_and_num_params_csv_path = 'depth_and_num_params.csv'
evaluate_csv_path = 'evaluate.csv'
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
merged_with_improvement.to_pickle("table_2.pkl")

print("Table 2 finished.")