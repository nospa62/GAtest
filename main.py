#
# Main Execution Script for OPF using Genetic Algorithm
#
# This script orchestrates the entire OPF analysis by:
# 1. Importing the necessary functions from the other modules.
# 2. Creating the CIGRE HV benchmark network.
# 3. Running an initial analysis on the base case.
# 4. Setting up and running the genetic algorithm to find the optimal solution.
# 5. Analyzing and reporting the final results in a comparative format.
# 6. Plotting the convergence of the genetic algorithm.
#
# To run this project, ensure all required files are in the same directory:
# - main.py (this file)
# - network_creator.py
# - opf_core.py
# - opf_analysis.py
# - adaptive_ga.py
# - ga_performance_analysis.py
#
# Dependencies:
# pip install geneticalgorithm2 pandapower numpy matplotlib
#

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga, AlgorithmParams
import copy

# --- Import project modules ---
try:
    from network_creator import create_cigre_hv_network
    from opf_core import objective_function, PENALTY_VALUE, TIME_INTERVAL_MIN
    from opf_analysis import analyze_and_report
    from ga_performance_analysis import plot_convergence, run_statistical_analysis
    # from adaptive_ga import adaptive_ga_callback # Temporarily disable adaptive GA
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required .py files are in the same directory.")
    exit()

# --- Main Execution Block ---
if __name__ == '__main__':

    # --- 1. Create the Network ---
    print("Creating the CIGRE HV Benchmark Network...")
    net = create_cigre_hv_network()
    print("Network created successfully.")

    # --- 2. Analyze the Initial State ---
    initial_results = analyze_and_report(net, "Initial Network State", time_interval_min=TIME_INTERVAL_MIN)

    # --- 3. Set up the Genetic Algorithm ---
    varbound = np.array(
        [[net.gen.min_p_mw.iloc[0], net.gen.max_p_mw.iloc[0]],
         [net.gen.min_p_mw.iloc[1], net.gen.max_p_mw.iloc[1]],
         [net.gen.min_p_mw.iloc[2], net.gen.max_p_mw.iloc[2]],
         [0.95, 1.05],
         [0.95, 1.05],
         [0.95, 1.05]]
    )

    seed_solution = np.array([
        net.gen.p_mw.iloc[0], net.gen.p_mw.iloc[1], net.gen.p_mw.iloc[2],
        net.gen.vm_pu.iloc[0], net.gen.vm_pu.iloc[1], net.gen.vm_pu.iloc[2]
    ])
    start_generation = np.array([seed_solution] * 20)

    params = AlgorithmParams(
        max_num_iteration=500,
        population_size=250,
        mutation_probability=0.2,
        elit_ratio=0.05,
        parents_portion=0.4,
        crossover_type='uniform',
        max_iteration_without_improv=150
        # callbacks=[adaptive_ga_callback] # Temporarily disable adaptive GA
    )

    model = ga(
        dimension=len(varbound),
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=params
    )

    print("\nRunning Genetic Algorithm for OPF on CIGRE HV Network...")
    print("This may take several minutes depending on population size and iterations.")

    obj_func_with_args = lambda x: objective_function(x, net, PENALTY_VALUE, TIME_INTERVAL_MIN)

    model.run(
        no_plot=False,
        function=obj_func_with_args,
        start_generation=start_generation,
        progress_bar_stream='stdout'
    )
    print("GA run complete.")

    # --- 4. Display Final Results and Comparison ---
    best_solution = model.result.variable
    min_cost = model.result.score

    if min_cost >= PENALTY_VALUE:
        print("\n" + "=" * 70)
        print("               OPTIMAL POWER FLOW RESULTS SUMMARY")
        print("=" * 70)
        print("\nWARNING: The algorithm did not find a feasible solution.")
        final_results = analyze_and_report(net, "Best (Infeasible) GA Solution", solution_vector=best_solution,
                                           time_interval_min=TIME_INTERVAL_MIN)
    else:
        net_final = copy.deepcopy(net)
        final_results = analyze_and_report(net_final, "Final Optimal Solution", solution_vector=best_solution,
                                           time_interval_min=TIME_INTERVAL_MIN)

    print("\n" + "=" * 70)
    print("                     BEFORE vs. AFTER OPTIMIZATION")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Initial State':<25} | {'Optimal State':<25}")
    print("-" * 70)
    print(f"{'Feasible':<25} | {str(initial_results['feasible']):<25} | {str(final_results['feasible']):<25}")
    print(
        f"{'Total Generation (MW)':<25} | {initial_results['total_gen_p']:>25,.2f} | {final_results['total_gen_p']:>25,.2f}")
    print(
        f"{'Total Load (MW)':<25} | {initial_results['total_load_p']:>25,.2f} | {final_results['total_load_p']:>25,.2f}")
    print(
        f"{'System Losses (MW)':<25} | {initial_results['total_loss_p']:>25,.2f} | {final_results['total_loss_p']:>25,.2f}")
    print(
        f"{'Total Generation Cost':<25} | ${initial_results['total_generation_cost']:>24,.2f} | ${final_results['total_generation_cost']:>24,.2f}")
    print(
        f"{'System Marginal Price (P)':<25} | ${initial_results['market_price_p']:>24,.2f} | ${final_results['market_price_p']:>24,.2f}")
    print(
        f"{'Total Consumer Cost (P)':<25} | ${initial_results['total_consumer_cost_p']:>24,.2f} | ${final_results['total_consumer_cost_p']:>24,.2f}")
    print("=" * 70)

    # --- 5. GA Performance Analysis ---
    # The convergence plot is shown automatically by the .run() command if no_plot=False

    # Optional: Run statistical analysis (can be time-consuming)
    # To run this, uncomment the following lines.
    # run_statistical_analysis(
    #     num_runs=5,
    #     function=lambda X: objective_function(X, net, PENALTY_VALUE, TIME_INTERVAL_MIN),
    #     varbound=varbound,
    #     params=params,
    #     start_gen=start_generation
    # )
