#
# Genetic Algorithm Performance Analysis Module
#
# This script contains functions to evaluate and visualize the performance
# of the genetic algorithm in solving the OPF problem.
#
# Dependencies:
# pip install geneticalgorithm2 numpy matplotlib
#

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga, AlgorithmParams
import matplotlib.pyplot as plt


def plot_convergence(model):
    """
    Plots the convergence of the genetic algorithm.

    Args:
        model: The trained geneticalgorithm2 model object.
    """
    if not hasattr(model, 'report') or not model.report:
        print("Model report not found. Cannot plot convergence.")
        return

    scores = model.report.best_scores
    generations = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(generations, scores, marker='.', linestyle='-', color='b')
    plt.title('Genetic Algorithm Convergence for OPF')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score (Cost)')
    plt.grid(True)
    plt.tight_layout()
    # Set y-axis to logarithmic scale to better visualize improvements
    # when penalties are present
    if any(score > 1000000 for score in scores):  # A rough check if penalties are dominant
        plt.yscale('log')
        plt.ylabel('Best Fitness Score (Cost) - Log Scale')

    plt.show()


def run_statistical_analysis(num_runs, function, varbound, params, start_gen):
    """
    Runs the genetic algorithm multiple times to gather performance statistics.

    Args:
        num_runs (int): The number of times to run the optimization.
        function (callable): The objective function to minimize.
        varbound (np.array): The variable boundaries.
        params (AlgorithmParams): The GA parameters.
        start_gen (np.array): The seed for the initial population.
    """
    print("\n" + "=" * 60)
    print(f"STARTING STATISTICAL ANALYSIS ({num_runs} RUNS)")
    print("=" * 60)

    best_scores = []

    for i in range(num_runs):
        print(f"\n--- Running Analysis Run {i + 1}/{num_runs} ---")
        model = ga(
            dimension=len(varbound),
            variable_type='real',
            variable_boundaries=varbound,
            algorithm_parameters=params
        )
        model.run(
            no_plot=True,
            function=function,
            start_generation=start_gen
        )
        best_scores.append(model.result.score)
        print(f"Run {i + 1} complete. Best score found: {model.result.score:,.2f}")

    best_scores = np.array(best_scores)

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Number of runs: {num_runs}")
    print(f"Mean Best Fitness:    ${np.mean(best_scores):,.2f}")
    print(f"Standard Deviation:   ${np.std(best_scores):,.2f}")
    print(f"Best Overall Score:   ${np.min(best_scores):,.2f}")
    print(f"Worst Overall Score:  ${np.max(best_scores):,.2f}")
    print("=" * 60)


if __name__ == '__main__':
    # This block is for testing this module independently.
    # It requires network_creator.py and opf_core.py to be in the same directory.
    print("--- Running ga_performance_analysis.py as a standalone test ---")
    try:
        from network_creator import create_cigre_hv_network
        from opf_core import objective_function, PENALTY_VALUE, TIME_INTERVAL_MIN

        # 1. Create the network
        test_net = create_cigre_hv_network()
        print("Network created successfully for testing.")

        # 2. Set up GA parameters for the test
        varbound = np.array(
            [[test_net.gen.min_p_mw.iloc[0], test_net.gen.max_p_mw.iloc[0]],
             [test_net.gen.min_p_mw.iloc[1], test_net.gen.max_p_mw.iloc[1]],
             [test_net.gen.min_p_mw.iloc[2], test_net.gen.max_p_mw.iloc[2]],
             [0.95, 1.05],
             [0.95, 1.05],
             [0.95, 1.05]]
        )
        seed_solution = np.array([
            test_net.gen.p_mw.iloc[0], test_net.gen.p_mw.iloc[1], test_net.gen.p_mw.iloc[2],
            test_net.gen.vm_pu.iloc[0], test_net.gen.vm_pu.iloc[1], test_net.gen.vm_pu.iloc[2]
        ])
        start_generation = np.array([seed_solution] * 20)

        # Use smaller parameters for a quick test run
        test_params = AlgorithmParams(
            max_num_iteration=50,
            population_size=50,
            mutation_probability=0.2,
            elit_ratio=0.05,
            parents_portion=0.4,
            crossover_type='uniform',
            max_iteration_without_improv=20
        )

        # 3. Run a single optimization to test plotting
        print("\n--- Testing convergence plot ---")
        single_run_model = ga(
            dimension=len(varbound),
            variable_type='real',
            variable_boundaries=varbound,
            algorithm_parameters=test_params
        )
        single_run_model.run(
            no_plot=True,
            function=lambda X: objective_function(X, test_net, PENALTY_VALUE, TIME_INTERVAL_MIN),
            start_generation=start_generation
        )
        plot_convergence(single_run_model)

        # 4. Run a small statistical analysis
        # Note: A real analysis should use a much higher num_runs (e.g., 30)
        run_statistical_analysis(
            num_runs=3,
            function=lambda X: objective_function(X, test_net, PENALTY_VALUE, TIME_INTERVAL_MIN),
            varbound=varbound,
            params=test_params,
            start_gen=start_generation
        )

    except ImportError:
        print("Could not import dependencies from `network_creator.py` or `opf_core.py`.")
        print("Please ensure these files are in the same folder to run this test.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")