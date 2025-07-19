import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from AP7.MultiStep import CustomNumericalMethods

"""Same code as in the AP6 just changing the call methods, instead of gaus_seldel and newtorn =, we call LMM methods"""
class RenewableEnergySystemEnhanced:
    def __init__(self):
        self.params = {
            'max_solar_capacity': 150,
            'max_wind_capacity': 120,
            'battery_capacity': 300,
            'grid_connection_limit': 100,
            'solar_efficiency': 0.20,
            'wind_efficiency': 0.38,
            'battery_charge_efficiency': 0.95,
            'battery_discharge_efficiency': 0.92,
            'temperature_sensitivity': 0.03,
            'wind_variability': 0.15,
        }

        self.generation_models = {
            'solar': lambda t, temp: (np.sin(2 * np.pi * t / 24) * 0.5 + 0.5) *
                                     (1 + self.params['temperature_sensitivity'] * temp),
            'wind': lambda t, wind_speed: (np.cos(2 * np.pi * t / 24) * 0.5 + 0.5) *
                                          (1 + self.params['wind_variability'] * wind_speed)
        }

        self.demand_model = {
            'residential': lambda t: 50 + 20 * np.sin(2 * np.pi * t / 24),
            'industrial': lambda t: 80 + 30 * np.cos(2 * np.pi * t / 24),
            'commercial': lambda t: 40 + 10 * np.sin(2 * np.pi * t / 12)
        }

    def system_dynamics(self, X, t, environmental_conditions):
        """System dynamics function - keeping original implementation"""
        solar_gen, wind_gen, battery_stored, grid_draw = X

        temp = environmental_conditions.get('temperature', 20)
        wind_speed = environmental_conditions.get('wind_speed', 5)

        solar_input = (self.params['max_solar_capacity'] *
                       self.params['solar_efficiency'] *
                       self.generation_models['solar'](t, temp))

        wind_input = (self.params['max_wind_capacity'] *
                      self.params['wind_efficiency'] *
                      self.generation_models['wind'](t, wind_speed))

        total_demand = (
                self.demand_model['residential'](t) +
                self.demand_model['industrial'](t) +
                self.demand_model['commercial'](t)
        )

        derivatives = np.array([
            solar_input * (1 - solar_gen / self.params['max_solar_capacity']) - solar_gen,
            wind_input * (1 - wind_gen / self.params['max_wind_capacity']) - wind_gen,
            (solar_gen + wind_gen - total_demand) * (
                0.9 if battery_stored < self.params['battery_capacity'] else 0
            ),
            min(max(total_demand - (solar_gen + wind_gen + battery_stored), 0),
                self.params['grid_connection_limit'])
        ])
        return derivatives

    def simulate_with_multiple_methods(self, T=24, dt=0.1):
        """Simulate using multiple numerical methods for comparison"""
        X0 = [10, 10, 50, 0]  # Initial state
        env_conditions = {'temperature': 20, 'wind_speed': 5}

        def system_wrapper(x, t):
            return self.system_dynamics(x, t, env_conditions)

        results = {}

        time_points, solution = CustomNumericalMethods.adams_bashforth_2(
            system_wrapper, X0, 0, T, dt
        )
        results['Adams-Bashforth 2'] = (time_points, solution)

        time_points, solution = CustomNumericalMethods.adams_bashforth_4(
            system_wrapper, X0, 0, T, dt
        )
        results['Adams-Bashforth 4'] = (time_points, solution)

        time_points, solution = CustomNumericalMethods.adams_moulton_2(
            system_wrapper, X0, 0, T, dt
        )
        results['Adams-Moulton 2'] = (time_points, solution)

        time_points, solution = CustomNumericalMethods.custom_dirk_method(
            system_wrapper, X0, 0, T, dt
        )
        results['DIRK'] = (time_points, solution)

        return results

    def visualize_method_comparison(self, results):
        """Create comprehensive visualization comparing different numerical methods"""
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, figure=fig)

        variables = ['Solar Generation', 'Wind Generation',
                     'Battery Storage', 'Grid Draw']
        colors = {'Adams-Bashforth 2': '#2ecc71',
                  'Adams-Bashforth 4': '#3498db',
                  'Adams-Moulton 2': '#e74c3c',
                  'DIRK': '#9b59b6'}

        line_styles = {'Adams-Bashforth 2': '--',
                       'Adams-Bashforth 4': ':',
                       'Adams-Moulton 2': '-.',
                       'DIRK': '-'}

        for idx, var in enumerate(variables):
            ax = fig.add_subplot(gs[idx])

            # Plot each method with different line style and offset
            for method, (t, sol) in results.items():
                # Add small offset to make overlapping lines visible
                offset = 0
                if method == 'Adams-Bashforth 2':
                    offset = 0.2
                elif method == 'Adams-Bashforth 4':
                    offset = -0.2
                elif method == 'Adams-Moulton 2':
                    offset = 0.1

                ax.plot(t, sol[:, idx] + offset,
                        label=method,
                        color=colors[method],
                        linestyle=line_styles[method],
                        linewidth=2,
                        alpha=0.8)

            ax.set_title(f'{var} Comparison Across Methods')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        return fig

    def compute_method_statistics(self, results):
        """Compute statistical comparisons between methods"""
        stats = {}
        reference_method = 'DIRK'  # Using DIRK as reference
        ref_time, ref_sol = results[reference_method]

        for method, (t, sol) in results.items():
            if method != reference_method:
                # Compute various error metrics
                abs_error = np.abs(sol - ref_sol)
                rel_error = abs_error / (np.abs(ref_sol) + 1e-10)  # Avoid division by zero

                stats[method] = {
                    'max_abs_error': np.max(abs_error),
                    'mean_abs_error': np.mean(abs_error),
                    'max_rel_error': np.max(rel_error),
                    'mean_rel_error': np.mean(rel_error)
                }

        return stats


def run_simulation_analysis():
    # Create system instance
    system = RenewableEnergySystemEnhanced()

    print("Running simulations with multiple numerical methods...")
    results = system.simulate_with_multiple_methods(T=24, dt=0.1)

    print("\nGenerating comparison visualizations...")
    system.visualize_method_comparison(results)
    plt.show()

    print("\nComputing method statistics...")
    stats = system.compute_method_statistics(results)

    print("\nMethod Comparison Statistics (relative to DIRK method):")
    for method, metrics in stats.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    run_simulation_analysis()