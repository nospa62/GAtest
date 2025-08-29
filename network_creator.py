#
# CIGRE High Voltage (HV) Benchmark Network Creator
#
# This script contains the function to programmatically build the CIGRE HV
# benchmark network based on detailed data.
#
# It is designed to be imported by other scripts that need to run analysis
# on this specific network model.
#
# Dependencies:
# pip install pandapower numpy
#

import pandapower as pp


def create_cigre_hv_network():
    """
    Creates and returns the CIGRE High Voltage benchmark network from a
    detailed data dictionary. This function is self-contained.

    Returns:
        pandapowerNet: A pandapower network object.
    """
    net_data = {
        'buses': [
            {'id': 0, 'name': 'Bus 1', 'vn_kv': 220.0},
            {'id': 1, 'name': 'Bus 2', 'vn_kv': 220.0},
            {'id': 2, 'name': 'Bus 3', 'vn_kv': 220.0},
            {'id': 3, 'name': 'Bus 4', 'vn_kv': 220.0},
            {'id': 4, 'name': 'Bus 5', 'vn_kv': 220.0},
            {'id': 5, 'name': 'Bus 6a', 'vn_kv': 220.0},
            {'id': 6, 'name': 'Bus 6b', 'vn_kv': 220.0},
            {'id': 7, 'name': 'Bus 7', 'vn_kv': 380.0},
            {'id': 8, 'name': 'Bus 8', 'vn_kv': 380.0},
            {'id': 9, 'name': 'Bus 9 (Slack)', 'vn_kv': 22.0},
            {'id': 10, 'name': 'Bus 10', 'vn_kv': 22.0},
            {'id': 11, 'name': 'Bus 11', 'vn_kv': 22.0},
            {'id': 12, 'name': 'Bus 12', 'vn_kv': 22.0}
        ],
        'loads': [
            {'bus': 1, 'p_mw': 225.0, 'q_mvar': 170.0},
            {'bus': 2, 'p_mw': 285.0, 'q_mvar': 194.0},
            {'bus': 3, 'p_mw': 296.0, 'q_mvar': 204.0},
            {'bus': 4, 'p_mw': 103.0, 'q_mvar': 42.0},
            {'bus': 5, 'p_mw': 365.0, 'q_mvar': 246.0}
        ],
        'gens': [
            {'id': 0, 'bus': 9, 'name': 'Gen 9 (Slack)', 'p_mw': 0, 'vm_pu': 1.03, 'min_p_mw': 0, 'max_p_mw': 2000, 'min_q_mvar': -1000, 'max_q_mvar': 1500},
            {'id': 1, 'bus': 10, 'name': 'Gen 10', 'p_mw': 400.0, 'vm_pu': 1.02, 'min_p_mw': 100, 'max_p_mw': 800,
             'min_q_mvar': -400, 'max_q_mvar': 500},
            {'id': 2, 'bus': 11, 'name': 'Gen 11', 'p_mw': 200.0, 'vm_pu': 1.02, 'min_p_mw': 50, 'max_p_mw': 450,
             'min_q_mvar': -200, 'max_q_mvar': 300},
            {'id': 3, 'bus': 12, 'name': 'Gen 12', 'p_mw': 150.0, 'vm_pu': 1.02, 'min_p_mw': 80, 'max_p_mw': 600,
             'min_q_mvar': -280, 'max_q_mvar': 350}
        ],
        'costs': [
            {'gen_id': 0, 'a': 0.008, 'b': 45, 'c': 100, 'd': 0.0020, 'e': 10},
            {'gen_id': 1, 'a': 0.012, 'b': 20, 'c': 80, 'd': 0.0015, 'e': 5},
            {'gen_id': 2, 'a': 0.010, 'b': 25, 'c': 90, 'd': 0.0018, 'e': 8},
        ],
        'lines': [
            {'from_bus': 0, 'to_bus': 1, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 100.0, 'max_i_ka': 1.14},
            {'from_bus': 0, 'to_bus': 5, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 300.0, 'max_i_ka': 1.14},
            {'from_bus': 1, 'to_bus': 4, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 300.0, 'max_i_ka': 1.14},
            {'from_bus': 2, 'to_bus': 3, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 100.0, 'max_i_ka': 1.14},
            {'from_bus': 2, 'to_bus': 3, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 100.0, 'max_i_ka': 1.14},  # Parallel line
            {'from_bus': 3, 'to_bus': 4, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 300.0, 'max_i_ka': 1.14},
            {'from_bus': 3, 'to_bus': 5, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 300.0, 'max_i_ka': 1.14},
            {'from_bus': 7, 'to_bus': 8, 'r_ohm_per_km': 0.0328, 'x_ohm_per_km': 0.312, 'c_nf_per_km': 11.50,
             'length_km': 600.0, 'max_i_ka': 1.32},
            {'from_bus': 5, 'to_bus': 6, 'r_ohm_per_km': 0.0653, 'x_ohm_per_km': 0.398, 'c_nf_per_km': 9.08,
             'length_km': 0.1, 'max_i_ka': 1.14},
        ],
        'trafos': [
            {'hv_bus': 7, 'lv_bus': 0, 'sn_mva': 1200.0, 'vn_hv_kv': 380.0, 'vn_lv_kv': 220.0, 'vk_percent': 13.0,
             'vkr_percent': 0.28, 'pfe_kw': 50, 'i0_percent': 0.06, 'shift_degree': 0.0},
            {'hv_bus': 8, 'lv_bus': 2, 'sn_mva': 1200.0, 'vn_hv_kv': 380.0, 'vn_lv_kv': 220.0, 'vk_percent': 13.0,
             'vkr_percent': 0.28, 'pfe_kw': 50, 'i0_percent': 0.06, 'shift_degree': 0.0},
            {'hv_bus': 0, 'lv_bus': 9, 'sn_mva': 2000.0, 'vn_hv_kv': 220.0, 'vn_lv_kv': 22.0, 'vk_percent': 13.0,
             'vkr_percent': 0.25, 'pfe_kw': 60, 'i0_percent': 0.07, 'shift_degree': 330.0},
            {'hv_bus': 1, 'lv_bus': 10, 'sn_mva': 1200.0, 'vn_hv_kv': 220.0, 'vn_lv_kv': 22.0, 'vk_percent': 13.0,
             'vkr_percent': 0.26, 'pfe_kw': 45, 'i0_percent': 0.08, 'shift_degree': 330.0},
            {'hv_bus': 2, 'lv_bus': 11, 'sn_mva': 1200.0, 'vn_hv_kv': 220.0, 'vn_lv_kv': 22.0, 'vk_percent': 13.0,
             'vkr_percent': 0.26, 'pfe_kw': 45, 'i0_percent': 0.08, 'shift_degree': 330.0},
            {'hv_bus': 6, 'lv_bus': 12, 'sn_mva': 1200.0, 'vn_hv_kv': 220.0, 'vn_lv_kv': 22.0, 'vk_percent': 13.0,
             'vkr_percent': 0.27, 'pfe_kw': 48, 'i0_percent': 0.08, 'shift_degree': 330.0},
        ]
    }

    net = pp.create_empty_network(name="CIGRE HV Transmission Benchmark")
    for bus in net_data['buses']:
        pp.create_bus(net, vn_kv=bus['vn_kv'], name=bus['name'], index=bus['id'])
    for load in net_data['loads']:
        pp.create_load(net, bus=load['bus'], p_mw=load['p_mw'], q_mvar=load['q_mvar'])

    for gen in net_data['gens']:
        if gen['name'] == 'Gen 9 (Slack)':
            pp.create_ext_grid(net, bus=gen['bus'], vm_pu=gen['vm_pu'], name=gen['name'],
                              min_p_mw=gen['min_p_mw'], max_p_mw=gen['max_p_mw'],
                              min_q_mvar=gen['min_q_mvar'], max_q_mvar=gen['max_q_mvar'])
        else:
            pp.create_gen(net, bus=gen['bus'], p_mw=gen['p_mw'], vm_pu=gen['vm_pu'],
                          min_p_mw=gen['min_p_mw'], max_p_mw=gen['max_p_mw'],
                          min_q_mvar=gen['min_q_mvar'], max_q_mvar=gen['max_q_mvar'],
                          name=gen['name'])
    for line in net_data['lines']:
        pp.create_line_from_parameters(net, from_bus=line['from_bus'], to_bus=line['to_bus'],
                                       length_km=line['length_km'], r_ohm_per_km=line['r_ohm_per_km'],
                                       x_ohm_per_km=line['x_ohm_per_km'], c_nf_per_km=line['c_nf_per_km'],
                                       max_i_ka=line['max_i_ka'])
    for trafo in net_data['trafos']:
        pp.create_transformer_from_parameters(net, hv_bus=trafo['hv_bus'], lv_bus=trafo['lv_bus'],
                                              sn_mva=trafo['sn_mva'], vn_hv_kv=trafo['vn_hv_kv'],
                                              vn_lv_kv=trafo['vn_lv_kv'], vk_percent=trafo['vk_percent'],
                                              vkr_percent=trafo['vkr_percent'], pfe_kw=trafo['pfe_kw'],
                                              i0_percent=trafo['i0_percent'], shift_degree=trafo['shift_degree'])

    net.gen_costs = {cost['gen_id']: cost for cost in net_data['costs']}

    net.slack_cost = {'a': 0.02, 'b': 60, 'c': 150, 'd': 0.01, 'e': 15}

    net.gen_adv_params = {
        0: {'max_s_mva': 1000, 'ramp_up_mw_min': 180.0, 'ramp_down_mw_min': 180.0},
        1: {'max_s_mva': 600, 'ramp_up_mw_min': 100.0, 'ramp_down_mw_min': 100.0},
        2: {'max_s_mva': 800, 'ramp_up_mw_min': 130.0, 'ramp_down_mw_min': 130.0},
    }

    net.slack_adv_params = {'ramp_up_mw_min': 280.0, 'ramp_down_mw_min': 280.0}

    net.prev_gen_p_mw = {
        0: 400.0,
        1: 300.0,
        2: 200.0,
    }
    net.prev_slack_p_mw = 500.0
    return net


if __name__ == '__main__':
    # This block allows for standalone testing of the network creation.
    print("--- Running network_creator.py as a standalone test ---")

    # 1. Create the network
    test_net = create_cigre_hv_network()
    print("Network created successfully.")

    # 2. Print a summary to verify
    print("\n" + "=" * 40)
    print("Network Summary:")
    print(f"  - Buses: {len(test_net.bus)}")
    print(f"  - Lines: {len(test_net.line)}")
    print(f"  - Transformers: {len(test_net.trafo)}")
    print(f"  - Loads: {len(test_net.load)}")
    print(f"  - Generators: {len(test_net.gen)}")
    print(f"  - External Grids: {len(test_net.ext_grid)}")
    print("=" * 40)

    # 3. Run a simple power flow to check for errors
    try:
        pp.runpp(test_net)
        print("\nInitial power flow successful.")
    except Exception as e:
        print(f"\nAn error occurred during the initial power flow check: {e}")