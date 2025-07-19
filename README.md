# Renewable Energy ODE Solver

Advanced numerical methods for solving ordinary differential equations in renewable energy system dynamics using Linear Multistep Methods (LMMs) and Diagonally Implicit Runge-Kutta (DIRK) methods.

## Overview

This project implements and analyzes various numerical methods for simulating the complex dynamics of renewable energy systems, including solar panels, wind turbines, battery storage, and grid interactions. The system handles inherent nonlinearities and multiple time scales present in real-world renewable energy applications.

## Mathematical Model

The renewable energy system is modeled as a four-dimensional dynamical system:

- **X₁**: Solar energy generation with capacity constraints
- **X₂**: Wind energy generation with capacity constraints  
- **X₃**: Battery storage dynamics with charging/discharging logic
- **X₄**: Grid interaction with demand balancing

The system equations capture:
- Solar and wind input variability
- Battery capacity limitations
- Grid demand balancing
- System efficiency factors

## Implemented Methods

### Linear Multistep Methods
- **Adams-Bashforth 2nd Order (AB2)**: Explicit method for standard conditions
- **Adams-Bashforth 4th Order (AB4)**: Higher-order explicit method for improved accuracy
- **Adams-Moulton 2nd Order (AM2)**: Implicit method with predictor-corrector approach

### Runge-Kutta Methods
- **Diagonally Implicit Runge-Kutta (DIRK)**: Superior performance for stiff systems near battery capacity limits

## Key Features

- **Error Estimation and Control**: Adaptive step size control for optimal accuracy
- **Stiff System Handling**: Specialized methods for near-capacity battery dynamics
- **Discontinuity Management**: Smoothing functions for battery capacity limits
- **Multiple Scenarios**: Base case, high variability, and stiff case testing

## Results Summary

Our numerical experiments demonstrate:

- **AB2/AB4**: Fast execution but potential instability near battery limits (minimal impact in optimized systems)
- **AM2**: Better stability with increased computational cost due to implicit nature
- **DIRK**: Superior performance in stiff regions, recommended for systems operating near capacity limits

## Installation

```bash
git clone https://github.com/yourusername/renewable-energy-ode-solver.git
cd renewable-energy-ode-solver
# Add installation commands for your specific implementation
```

## Usage

```bash
# Add usage examples based on your implementation
# Example:
# python simulate.py --method DIRK --scenario stiff
```

## Project Structure

```
renewable-energy-ode-solver/
├── README.md
├── src/
│   ├── methods/
│   │   ├── adams_bashforth.py
│   │   ├── adams_moulton.py
│   │   └── dirk.py
│   ├── models/
│   │   └── renewable_system.py
│   └── utils/
│       └── error_control.py
├── tests/
├── docs/
│   └── AP7.pdf
├── results/
└── requirements.txt
```

## Testing Scenarios

1. **Base Case**: Standard operating conditions with typical solar/wind variability
2. **High Variability**: Rapid environmental changes testing method robustness
3. **Stiff Case**: Near battery capacity limits challenging numerical stability

## Performance Analysis

The DIRK method shows the most consistent performance across all scenarios, particularly excelling in:
- Stiff dynamics near battery capacity limits
- Maintaining stability during rapid state changes
- Handling discontinuities in the system dynamics

## Dependencies

- NumPy
- SciPy
- Matplotlib (for visualization)
- [Add other dependencies based on your implementation]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement include:
- Additional numerical methods implementation
- Performance optimization
- Extended test scenarios
- Real-world data integration

## References

1. Hairer, E., Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*
2. Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential Equations*
3. Alexander, R. (1977). *Diagonally Implicit Runge-Kutta Methods for Stiff ODEs*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Kakhniashvili Terenti**

## Acknowledgments

- Advanced Numerical Methods course (AP 7)
- Research in renewable energy system modeling
- Numerical analysis community for method development
