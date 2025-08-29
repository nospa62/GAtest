#
# OPF Core Logic Module
#
# This script contains the core logic for the Optimal Power Flow problem,
# including the objective function that the genetic algorithm will minimize.
#
# It is designed to be imported by the main execution script.
#
# Dependencies:
# pip install pandapower numpy
#

import numpy as np
import pandapower as pp
import copy

# --- Global Constants ---
PENALTY_VALUE = 1e9
TIME_INTERVAL_MIN = 15


# --- Helper Function ---

def check_capability_curve(p_mw, q_mvar, gen_idx, net):
    """
    Checks if a generator's operating point is within its capability curve
    using a simplified, robust model based on nameplate ratings.
    """
    params = net.gen_adv_params[gen_idx]
    gen_table = net.gen.iloc[gen_idx]

    # 1. Check total apparent power against MVA rating
    s_mva_sq = p_mw ** 2 + q_mvar ** 2
    if s_mva_sq > params['max_s_mva'] ** 2:
        return False

    # 2. Check reactive power against min/max Q limits
    if not (gen_table.min_q_mvar <= q_mvar <= gen_table.max_q_mvar):
        return False

    return True


# --- Objective Function ---

def objective_function(X, net, penalty, time_interval):
    """
    The fitness function for the Genetic Algorithm.
    It evaluates a candidate solution 'X' and returns a single score.
    """
    net_copy = copy.deepcopy(net)
    total_penalty = 0.0

    # Unpack the solution vector X into the pandapower network
    for i in range(len(net_copy.gen)):
        net_copy.gen.loc[i, 'p_mw'] = X[i]
        net_copy.gen.loc[i, 'vm_pu'] = X[i + len(net_copy.gen)]

    # --- Run Load Flow ---
    try:
        pp.runpp(net_copy, algorithm='nr', tolerance_mva=1e-8, max_iteration=20)
    except pp.LoadflowNotConverged:
        return penalty * 10  # A higher penalty for non-convergence

    # --- Calculate Total Generation Cost ---
    total_cost = 0.0
    # Cost of controllable generators
    for i, gen in net_copy.res_gen.iterrows():
        p_mw, q_mvar = gen.p_mw, gen.q_mvar
        cost_params = net.gen_costs[i]
        total_cost += cost_params['a'] * p_mw ** 2 + cost_params['b'] * p_mw + cost_params['c']
        total_cost += cost_params['d'] * q_mvar ** 2 + cost_params['e'] * q_mvar

    # Add the cost of the slack generator
    slack_p_mw = net_copy.res_ext_grid.p_mw.iloc[0]
    slack_q_mvar = net_copy.res_ext_grid.q_mvar.iloc[0]
    slack_cost_params = net_copy.slack_cost
    total_cost += slack_cost_params['a'] * slack_p_mw ** 2 + slack_cost_params['b'] * slack_p_mw + slack_cost_params[
        'c']
    total_cost += slack_cost_params['d'] * slack_q_mvar ** 2 + slack_cost_params['e'] * slack_q_mvar

    # --- Flat Penalty Calculation ---
    for vm_pu in net_copy.res_bus.vm_pu:
        if not (0.95 <= vm_pu <= 1.05):
            total_penalty += penalty

    for loading in net_copy.res_line.loading_percent:
        if loading > 100:
            total_penalty += penalty

    for loading in net_copy.res_trafo.loading_percent:
        if loading > 100:
            total_penalty += penalty

    for i, gen in net_copy.res_gen.iterrows():
        p_mw_current, q_mvar_current = gen.p_mw, gen.q_mvar
        p_mw_prev = net.prev_gen_p_mw[i]
        params = net.gen_adv_params[i]
        max_ramp_up = params['ramp_up_mw_min'] * time_interval
        max_ramp_down = params['ramp_down_mw_min'] * time_interval
        if (p_mw_current - p_mw_prev > max_ramp_up) or (p_mw_prev - p_mw_current > max_ramp_down):
            total_penalty += penalty
        if not check_capability_curve(p_mw_current, q_mvar_current, i, net_copy):
            total_penalty += penalty

    return total_cost + total_penalty


if __name__ == '__main__':
    # This block is for testing this module independently.
    # It requires the network_creator.py file to be in the same directory.
    print("--- Running opf_core.py as a standalone test ---")
    try:
        from network_creator import create_cigre_hv_network

        # 1. Create the network
        test_net = create_cigre_hv_network()
        print("Network created successfully for testing.")

        # 2. Test the objective function with the initial state
        initial_solution = np.array([
            test_net.gen.p_mw.iloc[0], test_net.gen.p_mw.iloc[1], test_net.gen.p_mw.iloc[2],
            test_net.gen.vm_pu.iloc[0], test_net.gen.vm_pu.iloc[1], test_net.gen.vm_pu.iloc[2]
        ])

        initial_cost = objective_function(initial_solution, test_net, PENALTY_VALUE, TIME_INTERVAL_MIN)
        print(f"\nCost of initial state: ${initial_cost:,.2f}")

    except ImportError:
        print("Could not import `create_cigre_hv_network`.")
        print("Please ensure `network_creator.py` is in the same folder to run this test.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")

