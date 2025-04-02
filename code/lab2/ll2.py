import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numba import jit


class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization with Simplex and KKT")
        self.console_output = []

        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel on the right
        control_panel = tk.Frame(main_frame, width=200)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_panel, text="Initial X:").pack()
        self.x_entry = ttk.Entry(control_panel)
        self.x_entry.pack()
        self.x_entry.insert(0, "0")

        ttk.Label(control_panel, text="Initial Y:").pack()
        self.y_entry = ttk.Entry(control_panel)
        self.y_entry.pack()
        self.y_entry.insert(0, "0")

        ttk.Label(control_panel, text="Learning Rate:").pack()
        self.learning_rate_entry = ttk.Entry(control_panel)
        self.learning_rate_entry.pack()
        self.learning_rate_entry.insert(0, "0.01")

        ttk.Label(control_panel, text="Number of Iterations:").pack()
        self.max_iter_entry = ttk.Entry(control_panel)
        self.max_iter_entry.pack()
        self.max_iter_entry.insert(0, "100")

        ttk.Label(control_panel, text="Speed:").pack()
        self.timer_speed_edit = ttk.Entry(control_panel)
        self.timer_speed_edit.pack()
        self.timer_speed_edit.insert(0, "100")

        self.start_button = ttk.Button(
            control_panel, text="Start", command=self.start_optimization
        )
        self.start_button.pack(pady=10)

        # Matplotlib figure inside Tkinter
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.all_points = None
        self.current_point_index = 0
        self.point_timer = None

    def quadratic_function(self, x, y):
        return 2 * x**2 + 4 * x * y - 6 * x - 3 * y

    def set_task2(self):
        @jit(nopython=True)
        def quadratic_function(x, y):
            return 2 * x**2 + 4 * x * y - 6 * x - 3 * y

        @jit(nopython=True)
        def quadratic_gradient(x, y):
            df_dx = 4 * x + 4 * y - 6
            df_dy = 4 * x - 3
            return np.array([df_dx, df_dy])

        def is_within_constraints(x, y):
            return x >= 0 and y >= 0 and (x + y) <= 1 and (2 * x + 3 * y) <= 4

        def project_to_constraints(x, y):
            x = max(0, min(x, 1))
            y = max(0, min(y, 1))
            if (x + y) > 1:
                x, y = x / (x + y), y / (x + y)
            if (2 * x + 3 * y) > 4:
                scale = 4 / (2 * x + 3 * y)
                x, y = x * scale, y * scale
            return x, y

        self.max_iter = int(self.max_iter_entry.get())
        tolerance = 0.001
        self.timer_speed = int(self.timer_speed_edit.get())

        start_point = np.array([float(self.x_entry.get()), float(self.y_entry.get())])
        points = np.zeros((self.max_iter + 1, 2))
        points[0] = start_point
        current_point = start_point.copy()
        actual_iter = 0


        for i in range(self.max_iter):
            self.console_output.append(
                f"Текущая точка: {current_point} Значение:{quadratic_function(current_point[0], current_point[1]) }"
            )
            grad = quadratic_gradient(current_point[0], current_point[1])
            if np.linalg.norm(grad) < tolerance:
                break
            new_point = current_point - float(self.learning_rate_entry.get()) * grad
            new_point[0], new_point[1] = project_to_constraints(
                new_point[0], new_point[1]
            )
            points[i + 1] = new_point
            if np.linalg.norm(new_point - current_point) < tolerance:
                break
            current_point = new_point
            actual_iter = i + 1

        self.all_points = points[: actual_iter + 1]
        self.current_point_index = 0
        self.ax.clear()
        self.ax.set_title("Optimization Path")

        # Set up the function surface
        x_vals = np.linspace(0, 1, 100)
        y_vals = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = quadratic_function(X, Y)
        self.ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.7)

        # Initialize optimization path
        self.ax.plot([], [], [], "ko", markersize=5)
        self.ax.plot([], [], [], "ro", markersize=8)  # End point in red

        print(
            f"Локальный минимум найден в точке: {current_point}, Значение функции: {quadratic_function(current_point[0], current_point[1])}"
        )
        print(actual_iter)

        self.console_output.append(
            f"Локальный минимум найден в точке: {current_point}, Значение функции: {quadratic_function(current_point[0], current_point[1])}"
        )
        self.console_output.append(f"Количество итераций: {actual_iter}")

    # Update function to animate the optimization path
    def update(self, frame):
        if self.all_points is not None and len(self.all_points) > 0:
            x_vals = self.all_points[: frame + 1, 0]
            y_vals = self.all_points[: frame + 1, 1]
            z_vals = np.array(
                [self.quadratic_function(x, y) for x, y in zip(x_vals, y_vals)]
            )

            self.ax.plot(x_vals, y_vals, z_vals, "bo-", markersize=5)
            self.canvas.draw()

    # Start the optimization process and animation
    def start_optimization(self):
        self.set_task2()
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=self.timer_speed,
            blit=False,
        )
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
