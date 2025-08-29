#
# OPF Analysis and Reporting Module
#
# This script contains the functions necessary to analyze the results of a
# power flow run and generate a comprehensive report.
#
# Dependencies:
# pip install pandapower numpy
#

import numpy as np
import pandapower as pp
import copy


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


# --- Main Analysis and Reporting Function ---

def analyze_and_report(net_instance, case_name="Analysis Case", solution_vector=None, time_interval_min=15):
    """
    Runs a power flow on a given network state and prints a detailed
    feasibility and economic report. It also returns a dictionary of the results.
    """
    net_copy = copy.deepcopy(net_instance)
    if solution_vector is not None:
        for i in range(len(net_copy.gen)):
            net_copy.gen.loc[i, 'p_mw'] = solution_vector[i]
            net_copy.gen.loc[i, 'vm_pu'] = solution_vector[i + len(net_copy.gen)]

    results = {'case_name': case_name, 'feasible': True, 'violations': []}

    print("\n" + "=" * 60)
    print(f"DETAILED ANALYSIS FOR: {case_name}")
    print("=" * 60)
    try:
        pp.runpp(net_copy, algorithm='nr')
        print("Power flow converged successfully.")
        results['converged'] = True
    except pp.LoadflowNotConverged:
        print("!!! POWER FLOW FAILED TO CONVERGE !!!")
        results['converged'] = False
        results['feasible'] = False
        results['violations'].append("Power flow did not converge.")
        return results

    # --- Constraint Check ---
    print("\n--- Constraint Check ---")
    violations_found = False
    # Voltage Violations
    bad_voltages = net_copy.res_bus[(net_copy.res_bus.vm_pu < 0.95) | (net_copy.res_bus.vm_pu > 1.05)]
    if not bad_voltages.empty:
        violations_found = True
        results['violations'].append("Bus voltage limits violated.")
        print("\n--- VOLTAGE VIOLATIONS (limit: 0.95 - 1.05 p.u.) ---")
        print(bad_voltages.vm_pu)

    # Line Overloads
    bad_lines = net_copy.res_line[net_copy.res_line.loading_percent > 100]
    if not bad_lines.empty:
        violations_found = True
        results['violations'].append("Line loading limits violated.")
        print("\n--- LINE OVERLOADS (> 100%) ---")
        print(bad_lines.loading_percent)

    # Transformer Overloads
    bad_trafos = net_copy.res_trafo[net_copy.res_trafo.loading_percent > 100]
    if not bad_trafos.empty:
        violations_found = True
        results['violations'].append("Transformer loading limits violated.")
        print("\n--- TRANSFORMER OVERLOADS (> 100%) ---")
        print(bad_trafos.loading_percent)

    for i, gen in net_copy.res_gen.iterrows():
        p_mw_current, q_mvar_current = gen.p_mw, gen.q_mvar
        p_mw_prev = net_instance.prev_gen_p_mw[i]
        params = net_instance.gen_adv_params[i]
        max_ramp_up = params['ramp_up_mw_min'] * time_interval_min
        max_ramp_down = params['ramp_down_mw_min'] * time_interval_min
        if (p_mw_current - p_mw_prev > max_ramp_up):
            violations_found = True
            results['violations'].append(f"Ramp rate violation for Gen {i}.")
            print(f"\n--- RAMP-UP VIOLATION for Gen {i} ({net_instance.gen.name.iloc[i]}) ---")
            print(f"  Change: {p_mw_current - p_mw_prev:.2f} MW > Limit: {max_ramp_up:.2f} MW")
        if (p_mw_prev - p_mw_current > max_ramp_down):
            violations_found = True
            results['violations'].append(f"Ramp rate violation for Gen {i}.")
            print(f"\n--- RAMP-DOWN VIOLATION for Gen {i} ({net_instance.gen.name.iloc[i]}) ---")
            print(f"  Change: {p_mw_prev - p_mw_current:.2f} MW > Limit: {max_ramp_down:.2f} MW")

        if not check_capability_curve(p_mw_current, q_mvar_current, i, net_copy):
            violations_found = True
            results['violations'].append(f"Capability curve violation for Gen {i}.")
            print(f"\n--- CAPABILITY CURVE VIOLATION for Gen {i} ({net_instance.gen.name.iloc[i]}) ---")
            print(f"  Operating Point: P={p_mw_current:.2f} MW, Q={q_mvar_current:.2f} Mvar")

    if not violations_found:
        print(">>> No constraint violations found. The case is feasible. <<<")
        results['feasible'] = True
    else:
        print("\n>>> Constraint violations detected. The case is INFEASIBLE. <<<")
        results['feasible'] = False

    # --- Power Balance Summary ---
    total_gen_p = net_copy.res_gen.p_mw.sum() + net_copy.res_ext_grid.p_mw.sum()
    total_gen_q = net_copy.res_gen.q_mvar.sum() + net_copy.res_ext_grid.q_mvar.sum()
    total_load_p = net_copy.res_load.p_mw.sum()
    total_load_q = net_copy.res_load.q_mvar.sum()
    total_loss_p = total_gen_p - total_load_p
    total_loss_q = total_gen_q - total_load_q

    print("\n--- Power Balance Summary ---")
    print(f"Total Generation:   {total_gen_p:,.2f} MW, {total_gen_q:,.2f} Mvar")
    print(f"Total Consumption:  {total_load_p:,.2f} MW, {total_load_q:,.2f} Mvar")
    print(f"Total System Losses:{total_loss_p:,.2f} MW, {total_loss_q:,.2f} Mvar")

    # --- Economic Analysis ---
    print("\n--- Economic Analysis ---")
    total_gen_cost = 0
    marginal_costs_p = []
    marginal_costs_q = []

    # Controllable Generators
    for i, gen in net_copy.res_gen.iterrows():
        p_mw, q_mvar = gen.p_mw, gen.q_mvar
        cost_params = net_copy.gen_costs[i]
        cost_p = cost_params['a'] * p_mw ** 2 + cost_params['b'] * p_mw + cost_params['c']
        cost_q = cost_params['d'] * q_mvar ** 2 + cost_params['e'] * q_mvar
        total_cost_gen = cost_p + cost_q
        total_gen_cost += total_cost_gen
        marginal_cost_p = 2 * cost_params['a'] * p_mw + cost_params['b']
        marginal_cost_q = 2 * cost_params['d'] * q_mvar + cost_params['e']
        marginal_costs_p.append(marginal_cost_p)
        marginal_costs_q.append(marginal_cost_q)
        print(f"Dispatch & Cost for {net_copy.gen.name.iloc[i]}:")
        print(f"  - Dispatch (P, Q):    {p_mw:,.2f} MW, {q_mvar:,.2f} Mvar")
        print(f"  - Active Power Cost:  ${cost_p:,.2f} (Marginal: ${marginal_cost_p:.2f}/MWh)")
        print(f"  - Reactive Power Cost:${cost_q:,.2f} (Marginal: ${marginal_cost_q:.2f}/Mvarh)")
        print(f"  - Total Gen Cost:     ${total_cost_gen:,.2f}")

    # Slack Generator
    slack_p_mw = net_copy.res_ext_grid.p_mw.iloc[0]
    slack_q_mvar = net_copy.res_ext_grid.q_mvar.iloc[0]
    cost_params = net_copy.slack_cost
    cost_p = cost_params['a'] * slack_p_mw ** 2 + cost_params['b'] * slack_p_mw + cost_params['c']
    cost_q = cost_params['d'] * slack_q_mvar ** 2 + cost_params['e'] * slack_q_mvar
    total_cost_gen = cost_p + cost_q
    total_gen_cost += total_cost_gen
    marginal_cost_p = 2 * cost_params['a'] * slack_p_mw + cost_params['b']
    marginal_cost_q = 2 * cost_params['d'] * slack_q_mvar + cost_params['e']
    marginal_costs_p.append(marginal_cost_p)
    marginal_costs_q.append(marginal_cost_q)
    print(f"Dispatch & Cost for Slack Gen:")
    print(f"  - Dispatch (P, Q):    {slack_p_mw:,.2f} MW, {slack_q_mvar:,.2f} Mvar")
    print(f"  - Active Power Cost:  ${cost_p:,.2f} (Marginal: ${marginal_cost_p:.2f}/MWh)")
    print(f"  - Reactive Power Cost:${cost_q:,.2f} (Marginal: ${marginal_cost_q:.2f}/Mvarh)")
    print(f"  - Total Gen Cost:     ${total_cost_gen:,.2f}")

    # System Costs
    market_price_p = max(marginal_costs_p) if marginal_costs_p else 0
    market_price_q = max(marginal_costs_q) if marginal_costs_q else 0
    total_consumer_cost_p = market_price_p * total_load_p
    total_consumer_cost_q = market_price_q * total_load_q
    print("\n--- System Cost Summary ---")
    print(f"System Marginal Price (P): ${market_price_p:,.2f}/MWh")
    print(f"System Marginal Price (Q): ${market_price_q:,.2f}/Mvarh")
    print(f"Total Generation Cost:     ${total_gen_cost:,.2f}")
    print(f"Total Consumer Cost (P):   ${total_consumer_cost_p:,.2f}")
    print(f"Total Consumer Cost (Q):   ${total_consumer_cost_q:,.2f}")

    # --- Detailed Grid State ---
    print("\n--- Detailed Grid State ---")
    print("\nGrid State Summary:")
    print(f"  - Bus Voltages (Min/Max): {net_copy.res_bus.vm_pu.min():.3f} / {net_copy.res_bus.vm_pu.max():.3f} p.u.")
    print(f"  - Line Loading (Max):     {net_copy.res_line.loading_percent.max():.2f} %")
    print(f"  - Trafo Loading (Max):    {net_copy.res_trafo.loading_percent.max():.2f} %")

    print("\nFull Bus Results:")
    print(net_copy.res_bus)
    print("\nFull Line Loading:")
    print(net_copy.res_line.loading_percent)
    print("\nFull Transformer Loading:")
    print(net_copy.res_trafo.loading_percent)
    print("\nShunt State:")
    print(net_copy.res_shunt)

    print("=" * 60 + "\n")

    # Add key results to the dictionary for external use
    results.update({
        'total_gen_p': total_gen_p, 'total_load_p': total_load_p, 'total_loss_p': total_loss_p,
        'total_generation_cost': total_gen_cost, 'market_price_p': market_price_p, 'market_price_q': market_price_q,
        'total_consumer_cost_p': total_consumer_cost_p, 'total_consumer_cost_q': total_consumer_cost_q,
        'res_bus': net_copy.res_bus, 'res_line': net_copy.res_line, 'res_trafo': net_copy.res_trafo,
        'res_gen': net_copy.res_gen, 'res_ext_grid': net_copy.res_ext_grid
    })

    return results


if __name__ == '__main__':
    # This block is for testing this module independently.
    # It requires the network_creator.py file to be in the same directory.
    print("--- Running opf_analysis.py as a standalone test ---")
    try:
        from network_creator import create_cigre_hv_network

        # 1. Create the network
        test_net = create_cigre_hv_network()
        print("Network created successfully for testing.")

        # 2. Analyze the initial state
        initial_results = analyze_and_report(test_net, "Standalone Test: Initial State")

    except ImportError:
        print("Could not import `create_cigre_hv_network`.")
        print("Please ensure `network_creator.py` is in the same folder to run this test.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
