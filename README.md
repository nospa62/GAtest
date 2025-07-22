import numpy as np 
from geneticalgorithm import geneticalgorithm as ga 
import matplotlib.pyplot as plt 
import pandas as pd 
import warnings 
import time 
import json 

# Suppress warnings and set a random seed for reproducibility 
warnings.filterwarnings("ignore") 
np.random.seed(42) 

# Global lists to store data from the GA run for advanced reporting 
solution_history = [] 
final_population_fitness = [] 


# ===================================================================================== 
# --- 0. GA Callback Function --- 
# ===================================================================================== 
def on_gen(ga_instance): 
"""Callback function to record the best solution and final population fitness.""" 
solution_history.append(ga_instance.best_variable) 
# On the last generation, save the entire population's fitness for analysis 
if ga_instance.current_generation == ga_instance.max_num_iteration - 1: 
global final_population_fitness 
final_population_fitness = ga_instance.last_generation_fitness 


# ===================================================================================== 
# --- 1. Network Creation from User-Provided Data --- 
# ===================================================================================== 
def create_network_from_user_data(): 
"""Manually defines the network based on user data, with slack at Bus 9.""" 
print("Constructing network with slack bus at Bus 9...") 
base_mva = 1.0 

bus_data_raw = [(0, 220.0, '{"type":"Point","coordinates":[4,8.0]}'), 
(1, 220.0, '{"type":"Point","coordinates":[8,8.0]}'), 
(2, 220.0, '{"type":"Point","coordinates":[20,8.0]}'), 
(3, 220.0, '{"type":"Point","coordinates":[16,8.0]}'), 
(4, 220.0, '{"type":"Point","coordinates":[12,8.0]}'), 
(5, 220.0, '{"type":"Point","coordinates":[8,6.0]}'), 
(6, 220.0, '{"type":"Point","coordinates":[12,4.5]}'), 
(7, 380.0, '{"type":"Point","coordinates":[4,1.0]}'), 
(8, 380.0, '{"type":"Point","coordinates":[20,1.0]}'), 
(9, 22.0, '{"type":"Point","coordinates":[0,8.0]}'), 
(10, 22.0, '{"type":"Point","coordinates":[8,12.0]}'), 
(11, 22.0, '{"type":"Point","coordinates":[24,8.0]}'), 
(12, 22.0, '{"type":"Point","coordinates":[16,4.5]}'), 
(13, 22.0, '{"type":"Point","coordinates":[12,12]}')] 

load_data = [(1, 285, 200), (2, 325, 244), (3, 326, 244), (4, 103, 62), (5, 435, 296)] 
shunt_data = [(3, -160), (5, -180)] #(4, -80) 
original_gen_data = [(10, 500, 1.03), (11, 200, 1.03), (12, 300, 1.03)] 
line_data = [(0, 1, 100, 0.0653, 0.398, 1.14), (0, 5, 300, 0.0653, 0.398, 1.14), (1, 4, 300, 0.0653, 0.398, 1.14), 
(2, 3, 100, 0.0653, 0.398, 1.14), (2, 3, 100, 0.0653, 0.398, 1.14), (3, 4, 300, 0.0653, 0.398, 1.14), 
(3, 5, 300, 0.0653, 0.398, 1.14), (7, 8, 600, 0.0328, 0.312, 1.32), (5, 6, 0.1, 0.0653, 0.398, 1.14)] 
trafo_data = [(7, 0, 1000, 13, 0), (8, 2, 1000, 13, 0), (0, 9, 1000, 13, 0), 
(1, 10, 1000, 13, 0), (2, 11, 1000, 13, 0), (6, 12, 1000, 13, 0), (4, 13, 1000, 12, 0)] 

buses, bus_kv, bus_geodata = {}, {}, {} 
for bus_id, vn_kv, geo_str in bus_data_raw: 
buses[bus_id] = {'p_load': 0, 'q_load': 0, 'p_gen': 0, 'q_gen': 0, 'v_mag': 1.0, 'v_ang': 0.0, 'base_kv': vn_kv, 
'type': 3} 
bus_kv[bus_id], bus_geodata[bus_id] = vn_kv, json.loads(geo_str)['coordinates'] 
for bus_id, p, q in load_data: buses[bus_id]['p_load'] += p / base_mva; buses[bus_id]['q_load'] += q / base_mva 
for bus_id, q in shunt_data: buses[bus_id]['q_load'] += q / base_mva 

generators = {} 
print("Warning: Adding a slack generator at the new location: Bus 9.") 
generators[9] = {'p_max': 2000, 'p_min': -2000, 'v_set': 1.0, 'q_min': -1000, 'q_max': 1000, 'is_slack': True, 
'cost_a': 0.008, 'cost_b': 7, 'cost_c': 200, 'cost_d': 0.01, 'cost_e': 1.0} 
buses[9].update({'type': 1, 'v_mag': 1.0}) 

for bus_id, p_mw, vm_pu in original_gen_data: 
p_max = p_mw; 
generators[bus_id] = {'p_max': p_max, 'p_min': 0.0, 'v_set': vm_pu, 'q_min': -0.5 * p_max, 'q_max': 0.5 * p_max, 
'is_slack': False, 'cost_a': 0.012, 'cost_b': 10, 'cost_c': 100, 'cost_d': 0.015, 
'cost_e': 1.5} 
buses[bus_id].update({'type': 2, 'v_mag': vm_pu}) 

print("Adding a new synchronous condenser at Bus 13.") 
generators[13] = {'p_max': 0.0, 'p_min': 0.0, 'v_set': 1.0, 'q_min': -200, 'q_max': 300, 'is_slack': False, 
'cost_a': 0, 'cost_b': 0, 'cost_c': 50, 'cost_d': 0.005, 'cost_e': 0.8} 
buses[13].update({'type': 2, 'v_mag': 1.0}) 

branches = [] 
for f, t, l, r_km, x_km, i_ka in line_data: z_base = bus_kv[ 
f] ** 2 / base_mva; r, x = r_km * l / z_base, x_km * l / z_base; rating = np.sqrt( 
3) * bus_kv[f] * i_ka / base_mva; branches.append( 
{'from': f, 'to': t, 'r': r, 'x': x, 'rating': rating, 'type': 'Line'}) 
for h, l, s, vk, vkr in trafo_data: z_pu, r_pu = (vk / 100) * (base_mva / s), (vkr / 100) * ( 
base_mva / s); x_pu = np.sqrt(z_pu ** 2 - r_pu ** 2) if z_pu > r_pu else 0.0; branches.append( 
{'from': h, 'to': l, 'r': r_pu, 'x': x_pu, 'rating': s / base_mva, 'type': 'Trafo'}) 
return buses, branches, generators, base_mva, bus_geodata 


# ===================================================================================== 
# --- Core Simulation and Fitness Functions --- 
# ===================================================================================== 
def build_ybus(num_buses, branches): 
ybus = np.zeros((num_buses, num_buses), dtype=complex); 
for branch in branches: i, j = branch['from'], branch['to']; y = 1 / (branch['r'] + 1j * branch['x']); ybus[ 
i, j] -= y; ybus[j, i] -= y; ybus[i, i] += y; ybus[j, j] += y 
return ybus 


def solve_power_flow(buses, ybus, max_iter=30, tol=1e-6): 
local_buses = {k: v.copy() for k, v in buses.items()}; 
bus_ids_sorted = sorted(local_buses.keys()); 
num_buses = len(bus_ids_sorted) 
bus_map = {bus_id: i for i, bus_id in enumerate(bus_ids_sorted)} 
pq_idx = [bus_map[i] for i, b in local_buses.items() if b.get('type', 3) == 3]; 
pv_idx = [bus_map[i] for i, b in local_buses.items() if b.get('type', 3) == 2] 
non_slack_idx = sorted(pv_idx + pq_idx); 
v_mag = np.array([local_buses[i]['v_mag'] for i in bus_ids_sorted]); 
v_ang = np.array([local_buses[i]['v_ang'] for i in bus_ids_sorted]) 
converged = False 
for _ in range(max_iter): 
p_spec, q_spec = np.array( 
[local_buses[i]['p_gen'] - local_buses[i]['p_load'] for i in bus_ids_sorted]), np.array( 
[local_buses[i]['q_gen'] - local_buses[i]['q_load'] for i in bus_ids_sorted]) 
s_calc = (v_mag * np.exp(1j * v_ang)) * np.conj(ybus @ (v_mag * np.exp(1j * v_ang))); 
p_calc, q_calc = s_calc.real, s_calc.imag 
mismatch = np.concatenate((p_spec[non_slack_idx] - p_calc[non_slack_idx], q_spec[pq_idx] - q_calc[pq_idx])) 
if np.max(np.abs(mismatch)) < tol: converged = True; break 
J11, J12, J21, J22 = [np.zeros(s) for s in [(len(non_slack_idx),) * 2, (len(non_slack_idx), len(pq_idx)), 
(len(pq_idx), len(non_slack_idx)), (len(pq_idx),) * 2]] 
for i_idx, i in enumerate(non_slack_idx): 
for j_idx, j in enumerate(non_slack_idx): J11[i_idx, j_idx] = v_mag[i] * v_mag[j] * ( 
ybus[i, j].real * np.sin(v_ang[i] - v_ang[j]) - ybus[i, j].imag * np.cos( 
v_ang[i] - v_ang[j])) if i != j else -q_calc[i] - v_mag[i] ** 2 * ybus[i, i].imag 
for j_idx, j in enumerate(pq_idx): J12[i_idx, j_idx] = v_mag[i] * ( 
ybus[i, j].real * np.cos(v_ang[i] - v_ang[j]) + ybus[i, j].imag * np.sin( 
v_ang[i] - v_ang[j])) if i != j else p_calc[i] / v_mag[i] + v_mag[i] * ybus[i, i].real 
for i_idx, i in enumerate(pq_idx): 
for j_idx, j in enumerate(non_slack_idx): J21[i_idx, j_idx] = -v_mag[i] * v_mag[j] * ( 
ybus[i, j].real * np.cos(v_ang[i] - v_ang[j]) + ybus[i, j].imag * np.sin( 
v_ang[i] - v_ang[j])) if i != j else p_calc[i] - v_mag[i] ** 2 * ybus[i, i].real 
for j_idx, j in enumerate(pq_idx): J22[i_idx, j_idx] = v_mag[i] * ( 
ybus[i, j].real * np.sin(v_ang[i] - v_ang[j]) - ybus[i, j].imag * np.cos( 
v_ang[i] - v_ang[j])) if i != j else q_calc[i] / v_mag[i] - v_mag[i] * ybus[i, i].imag 
J = np.block([[J11, J12], [J21, J22]]) 
try: 
corr = np.linalg.solve(J, mismatch); v_ang[non_slack_idx] += corr[:len(non_slack_idx)]; v_mag[ 
pq_idx] += corr[len(non_slack_idx):] 
except np.linalg.LinAlgError: 
return local_buses, False 
if converged: 
s_final = (v_mag * np.exp(1j * v_ang)) * np.conj(ybus @ (v_mag * np.exp(1j * v_ang))) 
for i, bus_id in enumerate(bus_ids_sorted): local_buses[bus_id].update( 
{'v_mag_final': v_mag[i], 'v_ang_final': v_ang[i], 
'p_gen_final': s_final.real[i] + local_buses[bus_id]['p_load'], 
'q_gen_final': s_final.imag[i] + local_buses[bus_id]['q_load']}) 
return local_buses, converged 


def f(X): 
current_buses = {k: v.copy() for k, v in network_buses.items()}; 
controlled_gens = [g for g in network_gens.values() if not g['is_slack']]; 
x_idx = 0 
for gen in controlled_gens: 
bus_id = [k for k, v in network_gens.items() if v == gen][0] 
if gen['p_max'] > 0: current_buses[bus_id]['p_gen'] = X[x_idx] / base_mva; x_idx += 1 
current_buses[bus_id]['v_mag'] = X[x_idx]; 
x_idx += 1 
current_buses[[k for k, v in network_gens.items() if v['is_slack']][0]]['v_mag'] = X[-1] 
solved_buses, converged = solve_power_flow(current_buses, ybus) 
if not converged: return 1e8 
total_cost, penalty = 0, 0 
for bus_id, gen in network_gens.items(): 
p_mw, q_mvar = solved_buses[bus_id]['p_gen_final'] * base_mva, solved_buses[bus_id]['q_gen_final'] * base_mva 
total_cost += (gen['cost_a'] * p_mw ** 2 + gen['cost_b'] * p_mw + gen['cost_c']) + ( 
gen['cost_d'] * q_mvar ** 2 + gen['cost_e'] * q_mvar) 
if q_mvar < gen['q_min'] or q_mvar > gen['q_max']: penalty += 1e6 
if gen['is_slack'] and ( 
(p_slack := solved_buses[bus_id]['p_gen_final'] * base_mva) < gen['p_min'] or p_slack > gen[ 
'p_max']): penalty += 1e6 
for b in solved_buses.values(): 
v = b['v_mag_final']; 
if v < 0.95 or v > 1.05: penalty += 1e6 * (min(abs(v - 0.95), abs(v - 1.05))) ** 2 
for branch in network_branches: 
limit = 0.90 * branch['rating']; 
i, j = branch['from'], branch['to'] 
vi, vj = solved_buses[i]['v_mag_final'] * np.exp(1j * solved_buses[i]['v_ang_final']), solved_buses[j][ 
'v_mag_final'] * np.exp(1j * solved_buses[j]['v_ang_final']) 
s_ij = vi * np.conj((vi - vj) / (branch['r'] + 1j * branch['x'])) 
if np.abs(s_ij) > limit and limit > 0: penalty += 1e6 * (np.abs(s_ij) - limit) ** 2 
return total_cost + penalty 


# ===================================================================================== 
# --- 4. NEW and ENHANCED Analysis and Reporting Functions --- 
# ===================================================================================== 
def generate_ga_report(model, params, start_time, end_time, final_pop_fitness): 
"""Prints a detailed report of GA performance and parameters.""" 
print("\n\n" + "=" * 50) 
print(" GENETIC ALGORITHM PERFORMANCE REPORT") 
print("=" * 50) 

# --- Parameter Summary --- 
print("\n[GA Parameter Summary]") 
param_df = pd.DataFrame.from_dict(params, orient='index', columns=['Value']) 
print(param_df) 

# --- Performance Metrics --- 
print("\n[GA Performance Metrics]") 
initial_fitness = model.report[0] 
final_fitness = model.report[-1] 
total_improvement = initial_fitness - final_fitness 
generations_run = len(model.report) 
metrics_data = { 
'Runtime (sec)': f"{end_time - start_time:.2f}", 
'Generations Run': generations_run, 
'Initial Best Fitness ($/hr)': f"{initial_fitness:,.2f}", 
'Final Best Fitness ($/hr)': f"{final_fitness:,.2f}", 
'Total Improvement ($/hr)': f"{total_improvement:,.2f}", 
'Avg. Improvement per Gen': f"{total_improvement / generations_run:,.2f}" 
} 
print(pd.DataFrame.from_dict(metrics_data, orient='index', columns=['Value'])) 

# --- Final Population Statistics --- 
if final_pop_fitness: 
print("\n[Final Population Fitness Statistics]") 
stats_data = { 
'Best Fitness': f"{np.min(final_pop_fitness):,.2f}", 
'Worst Fitness': f"{np.max(final_pop_fitness):,.2f}", 
'Mean Fitness': f"{np.mean(final_pop_fitness):,.2f}", 
'Std. Deviation': f"{np.std(final_pop_fitness):,.2f}" 
} 
print(pd.DataFrame.from_dict(stats_data, orient='index', columns=['Value'])) 

# Fitness Histogram 
plt.figure(figsize=(10, 6)) 
plt.hist(final_pop_fitness, bins=20, color='teal', edgecolor='black') 
plt.title('Histogram of Final Population Fitness', fontsize=16) 
plt.xlabel('Fitness (Cost in $/hr)') 
plt.ylabel('Number of Individuals') 
plt.grid(axis='y', alpha=0.75) 
plt.show() 


def generate_opf_report(solved_buses, network_branches, network_gens, base_mva, best_fitness): 
"""Prints a detailed report of the final OPF solution.""" 
print("\n\n" + "=" * 50) 
print(" OPTIMAL POWER FLOW RESULTS") 
print("=" * 50) 

# --- System Losses --- 
p_gen_total = sum(b['p_gen_final'] for b in solved_buses.values()) * base_mva 
p_load_total = sum(b['p_load'] for b in solved_buses.values()) * base_mva 
q_gen_total = sum(b['q_gen_final'] for b in solved_buses.values()) * base_mva 
q_load_total = sum(b['q_load'] for b in solved_buses.values()) * base_mva 
p_loss, q_loss = p_gen_total - p_load_total, q_gen_total - q_load_total 
print("\n[Total System Losses]") 
print(f" - Active Power Loss: {p_loss:.4f} MW") 
print(f" - Reactive Power Loss: {q_loss:.4f} MVAr") 

# --- Generator Dispatch & Cost --- 
print(f"\n🏆 Minimum Generation Cost (P+Q): ${best_fitness:,.2f} / hour") 
gen_report = [] 
for bus_id, g in sorted(network_gens.items()): 
p, q, v = solved_buses[bus_id]['p_gen_final'] * base_mva, solved_buses[bus_id]['q_gen_final'] * base_mva, \ 
solved_buses[bus_id]['v_mag_final'] 
p_cost = g['cost_a'] * p ** 2 + g['cost_b'] * p + g['cost_c'] 
q_cost = g['cost_d'] * q ** 2 + g['cost_e'] * q 
gen_report.append( 
{'Bus': bus_id, 'Type': "Slack" if g['is_slack'] else ("Sync Cond" if g['p_max'] == 0 else "PV Gen"), 
'P Gen (MW)': p, 'Q Gen (MVAr)': q, 'V Set (pu)': v, 'P Cost': p_cost, 'Q Cost': q_cost, 
'Total Cost': p_cost + q_cost}) 
print("\n[Generator Dispatch & Cost Breakdown]") 
print(pd.DataFrame(gen_report).round(2).to_string(index=False)) 

# --- Bus Summary Table --- 
bus_report = [{'Bus': id, 'V (pu)': b['v_mag_final'], 'Angle (deg)': np.rad2deg(b['v_ang_final']), 
'P Gen (MW)': b['p_gen_final'] * base_mva, 'Q Gen (MVAr)': b['q_gen_final'] * base_mva, 
'P Load (MW)': b['p_load'] * base_mva, 'Q Load (MVAr)': b['q_load'] * base_mva} for id, b in 
sorted(solved_buses.items())] 
print("\n[Bus Summary Table]") 
print(pd.DataFrame(bus_report).round(3).to_string(index=False)) 

# --- Branch Flow & Loss Table --- 
branch_report = [] 
for b in network_branches: 
i, j, y_ij = b['from'], b['to'], 1 / (b['r'] + 1j * b['x']) 
vi, vj = solved_buses[i]['v_mag_final'] * np.exp(1j * solved_buses[i]['v_ang_final']), solved_buses[j][ 
'v_mag_final'] * np.exp(1j * solved_buses[j]['v_ang_final']) 
s_ij, s_ji = vi * np.conj((vi - vj) * y_ij), vj * np.conj((vj - vi) * y_ij) 
s_loss = (s_ij + s_ji) * base_mva 
flow_mva, limit_mva = np.abs(s_ij) * base_mva, b['rating'] * base_mva * 0.90 
loading = (flow_mva / limit_mva) * 100 if limit_mva > 0 else 0 
branch_report.append({'From': i, 'To': j, 'Type': b['type'], 'Flow (MVA)': flow_mva, 'Limit (MVA)': limit_mva, 
'Loading (%)': loading, 'P Loss (MW)': s_loss.real, 'Q Loss (MVAr)': s_loss.imag}) 
print("\n[Branch Flow & Loss Analysis (90% Limit)]") 
print(pd.DataFrame(branch_report).round(2).to_string(index=False)) 


def plot_voltage_profile(solved_buses): 
"""Creates a bar chart of the final bus voltage profile.""" 
bus_ids, voltages = sorted(solved_buses.keys()), [solved_buses[i]['v_mag_final'] for i in 
sorted(solved_buses.keys())] 
plt.figure(figsize=(14, 7)); 
bars = plt.bar([str(i) for i in bus_ids], voltages, color='c') 
plt.axhline(y=1.05, color='r', linestyle='--', label='Max Limit (1.05 pu)'); 
plt.axhline(y=0.95, color='r', linestyle='--', label='Min Limit (0.95 pu)') 
plt.title('Final Bus Voltage Profile', fontsize=18); 
plt.xlabel('Bus Number', fontsize=12); 
plt.ylabel('Voltage (pu)', fontsize=12) 
plt.ylim(0.9, 1.1); 
plt.legend(); 
plt.grid(axis='y', linestyle=':') 
for bar in bars: plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.4f}', 
ha='center', va='bottom', color='black') 
plt.show() 


def plot_network_topology(buses, branches, generators, coords): 
"""Creates a plot of the network topology.""" 
gen_buses = list(generators.keys()); 
load_buses = [i for i, b in buses.items() if b['p_load'] > 0 and i not in gen_buses] 
plt.figure(figsize=(16, 10)); 
for branch in branches: p1, p2 = coords[branch['from']], coords[branch['to']]; color = 'purple' if branch[ 
'type'] == 'Trafo' else 'gray'; plt.plot( 
[p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.0, zorder=1) 
node_size = 300 
plt.scatter([c[0] for i, c in coords.items() if i not in gen_buses and i not in load_buses], 
[c[1] for i, c in coords.items() if i not in gen_buses and i not in load_buses], s=node_size, 
c='skyblue', zorder=2, label='Bus') 
plt.scatter([c[0] for i, c in coords.items() if i in load_buses], 
[c[1] for i, c in coords.items() if i in load_buses], s=node_size, c='gold', zorder=2, label='Load Bus') 
plt.scatter([c[0] for i, c in coords.items() if i in gen_buses], 
[c[1] for i, c in coords.items() if i in gen_buses], s=node_size * 1.5, c='tomato', marker='s', 
zorder=2, label='Generator Bus') 
for i, pos in coords.items(): plt.text(pos[0], pos[1] + 0.3, str(i), ha='center', va='bottom', fontsize=12, 
fontweight='bold') 
plt.title('CIGRE HV Network Topology', fontsize=20); 
plt.legend(fontsize=12, loc='best'); 
plt.grid(True, linestyle='--', alpha=0.5); 
plt.xlabel('X Coordinate'); 
plt.ylabel('Y Coordinate'); 
plt.show() 


# ===================================================================================== 
# --- Main Execution Block --- 
# ===================================================================================== 
if __name__ == '__main__': 
network_buses, network_branches, network_gens, base_mva, bus_geodata = create_network_from_user_data() 
num_buses = max(network_buses.keys()) + 1 
ybus = build_ybus(num_buses, network_branches) 
print("✅ Network model created from user data.") 

varbound = [] 
non_slack_gens = [g for g in network_gens.values() if not g['is_slack']] 
for gen in non_slack_gens: 
if gen['p_max'] > 0: varbound.append([gen['p_min'], gen['p_max']]) 
varbound.append([0.95, 1.05]) 
varbound.append([0.95, 1.05]); 
varbound = np.array(varbound, dtype=object) 

algorithm_param = {'max_num_iteration': 200, 'population_size': 60, 'mutation_probability': 0.1, 'elit_ratio': 0.02, 
'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 
'max_iteration_without_improv': 20} 

model = ga(function=f, dimension=len(varbound), variable_type='real', variable_boundaries=varbound, 
algorithm_parameters=algorithm_param) 

print("\n🧬 Running GA for HV OPF...") 
start_time = time.time(); 
model.run(); 
end_time = time.time() 

generate_ga_report(model, algorithm_param, start_time, end_time, final_population_fitness) 

final_buses, x_idx = {k: v.copy() for k, v in network_buses.items()}, 0 
for gen in non_slack_gens: 
bus_id = [k for k, v in network_gens.items() if v == gen][0] 
if gen['p_max'] > 0: final_buses[bus_id]['p_gen'] = model.best_variable[x_idx] / base_mva; x_idx += 1 
final_buses[bus_id]['v_mag'] = model.best_variable[x_idx]; 
x_idx += 1 
final_buses[[k for k, v in network_gens.items() if v['is_slack']][0]]['v_mag'] = model.best_variable[-1] 
final_solved, final_converged = solve_power_flow(final_buses, ybus) 

if final_converged: 
generate_opf_report(final_solved, network_branches, network_gens, base_mva, model.best_function) 
plot_voltage_profile(final_solved) 
plot_network_topology(network_buses, network_branches, network_gens, bus_geodata) 
else: 
print("\n⚠️ Power flow did not converge with the GA's final solution.") 
