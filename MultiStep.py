import numpy as np
import scipy.optimize as optimize


class CustomNumericalMethods:
    @staticmethod
    def adams_bashforth_2(f, x0, t0, tf, h):
        """
        2-step Adams-Bashforth method (2nd order)

        Args:
        f: Function that returns derivative (right-hand side of ODE)
        x0: Initial state vector
        t0: Initial time
        tf: Final time
        h: Step size

        Returns:
        time_points: Array of time points
        solution: Array of solution values at each time point
        """
        # Calculate number of steps
        num_steps = int((tf - t0) / h) + 1
        time_points = np.linspace(t0, tf, num_steps)

        # Initialize solution array
        solution = np.zeros((num_steps, len(x0)))
        solution[0] = x0

        # First step using Euler method to for the second k, after that it will take off on its own
        if num_steps > 1:
            k1 = f(x0, t0)
            x1 = x0 + h * k1
            solution[1] = x1

            # we have already 
            for i in range(2, num_steps):
                t = time_points[i]
                k1 = f(solution[i - 1], time_points[i - 1])
                k0 = f(solution[i - 2], time_points[i - 2])

                x_next = solution[i - 1] + h * (1.5 * k1 - 0.5 * k0)
                solution[i] = x_next

        return time_points, solution

    @staticmethod
    def adams_moulton_2(f, x0, t0, tf, h):
        """
        2-step Adams-Moulton method (2nd order implicit method)

        Args:
        f: Function that returns derivative (right-hand side of ODE)
        x0: Initial state vector
        t0: Initial time
        tf: Final time
        h: Step size

        Returns:
        time_points: Array of time points
        solution: Array of solution values at each time point
        """
        num_steps = int((tf - t0) / h) + 1
        time_points = np.linspace(t0, tf, num_steps)
        solution = np.zeros((num_steps, len(x0)))
        solution[0] = x0
        if num_steps > 1:
            k1 = f(x0, t0)
            x1 = x0 + h * k1
            solution[1] = x1

            for i in range(2, num_steps):
                t = time_points[i]

                def implicit_equation(x_next):
                    """
                    Implicit equation to solve for next step
                    Uses predictor-corrector approach
                    """
                    # Predictor (Adams-Bashforth)
                    k1 = f(solution[i - 1], time_points[i - 1])
                    k0 = f(solution[i - 2], time_points[i - 2])
                    x_pred = solution[i - 1] + h * (1.5 * k1 - 0.5 * k0)

                    k_next = f(x_next, t)
                    k1 = f(solution[i - 1], time_points[i - 1])
                    return x_next - (solution[i - 1] + h * (0.5 * k_next + 0.5 * k1))
                solution[i] = optimize.fsolve(implicit_equation, solution[i - 1])

        return time_points, solution


#basically RK4 on steroids
    @staticmethod
    def adams_bashforth_4(f, x0, t0, tf, h):
        """
        4-step Adams-Bashforth method (4th order)

        Args:
        f: Function that returns derivative (right-hand side of ODE)
        x0: Initial state vector
        t0: Initial time
        tf: Final time
        h: Step size

        Returns:
        time_points: Array of time points
        solution: Array of solution values at each time point
        """
        num_steps = int((tf - t0) / h) + 1
        time_points = np.linspace(t0, tf, num_steps)

        solution = np.zeros((num_steps, len(x0)))
        solution[0] = x0

        if num_steps > 1:
            for i in range(1, min(4, num_steps)):
                t = time_points[i - 1]
                k1 = f(solution[i - 1], t)
                k2 = f(solution[i - 1] + 0.5 * h * k1, t + 0.5 * h)
                k3 = f(solution[i - 1] + 0.5 * h * k2, t + 0.5 * h)
                k4 = f(solution[i - 1] + h * k3, t + h)

                solution[i] = solution[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            for i in range(4, num_steps):
                t = time_points[i]
                k = [f(solution[i - j], time_points[i - j]) for j in range(1, 5)]
                x_next = solution[i - 1] + h * (
                        55 / 24 * k[0] -
                        59 / 24 * k[1] +
                        37 / 24 * k[2] -
                        9 / 24 * k[3]
                )
                solution[i] = x_next

        return time_points, solution

    @staticmethod
    def custom_dirk_method(f, x0, t0, tf, h):
        """
        Diagonally Implicit Runge-Kutta (DIRK) method

        Args:
        f: Function that returns derivative (right-hand side of ODE)
        x0: Initial state vector
        t0: Initial time
        tf: Final time
        h: Step size

        Returns:
        time_points: Array of time points
        solution: Array of solution values at each time point
        """
        num_steps = int((tf - t0) / h) + 1
        time_points = np.linspace(t0, tf, num_steps)

        solution = np.zeros((num_steps, len(x0)))
        solution[0] = x0

        for i in range(1, num_steps):
            t_prev = time_points[i - 1]
            t_curr = time_points[i]

            def implicit_equation(x_next):
                """
                Solve the implicit equation for the next step
                Uses Radau IIA method coefficients
                """
                gamma = (3 - np.sqrt(3)) / 6

                def stage_equation(k1):
                    stage_arg = solution[i - 1] + h * gamma * k1
                    return f(stage_arg, t_prev + h * gamma) - k1

                k1 = optimize.fsolve(stage_equation, f(solution[i - 1], t_prev))

                return solution[i - 1] + h * k1 - x_next

            solution[i] = optimize.fsolve(implicit_equation, solution[i - 1])

        return time_points, solution
