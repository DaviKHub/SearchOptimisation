import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from controller.optimizer_controller import OptimizerController
from model.himmelblau import himmelblau
from model.quadratic_task import quadratic_function

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Оптимизация")

        self.controller = OptimizerController()

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_panel = tk.Frame(main_frame)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.entries = {}
        for label, default in [
            ("Initial X:", "0"),
            ("Initial Y:", "0"),
            ("Step Size:", "0.01"),
            ("Number of Iterations:", "100"),
            ("Animation Speed:", "100"),
        ]:
            ttk.Label(control_panel, text=label).pack()
            entry = ttk.Entry(control_panel)
            entry.pack()
            entry.insert(0, default)
            self.entries[label] = entry

        ttk.Label(control_panel, text="Optimizer:").pack()
        self.optimizer_choice = ttk.Combobox(
            control_panel, values=self.controller.get_available_optimizers()
        )
        self.optimizer_choice.set("Gradient Descent")
        self.optimizer_choice.pack()

        self.start_button = ttk.Button(
            control_panel, text="Start", command=self.start_optimization
        )
        self.start_button.pack(pady=10)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.all_points = None
        self.timer_speed = 100

    def start_optimization(self):
        x0 = float(self.entries["Initial X:"].get())
        y0 = float(self.entries["Initial Y:"].get())
        step = float(self.entries["Step Size:"].get())
        iterations = int(self.entries["Number of Iterations:"].get())
        self.timer_speed = int(self.entries["Animation Speed:"].get())

        optimizer_name = self.optimizer_choice.get()
        optimizer = self.controller.get_optimizer(optimizer_name, step, iterations)

        self.all_points = optimizer.optimize(x0, y0)
        self.animate_path()

    def animate_path(self):
        self.ax.clear()
        if self.optimizer_choice.get() == "Gradient Descent":
            x_vals = np.linspace(-5, 5, 100)
            y_vals = np.linspace(-5, 5, 100)
            Z_func = himmelblau
        else:
            x_vals = np.linspace(0, 1, 100)
            y_vals = np.linspace(0, 1, 100)
            Z_func = quadratic_function

        X, Y = np.meshgrid(x_vals, y_vals)
        Z = Z_func(X, Y)

        self.ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.7)
        self.ax.set_title("")

        path = self.all_points
        (scat,) = self.ax.plot([], [], [], "ko", markersize=5)
        (red_dot,) = self.ax.plot([], [], [], "ro", markersize=8)

        def update(frame):
            if path.shape[1] == 3:
                # Path already contains x, y, z
                scat.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
                scat.set_3d_properties(path[: frame + 1, 2])
                red_dot.set_data([path[frame, 0]], [path[frame, 1]])
                red_dot.set_3d_properties([path[frame, 2]])
            else:
                # Path contains only x, y, need to compute z
                x_vals = path[: frame + 1, 0]
                y_vals = path[: frame + 1, 1]
                z_vals = np.array([Z_func(x, y) for x, y in zip(x_vals, y_vals)])

                scat.set_data(x_vals, y_vals)
                scat.set_3d_properties(z_vals)
                red_dot.set_data([x_vals[-1]], [y_vals[-1]])
                red_dot.set_3d_properties([z_vals[-1]])
            return scat, red_dot

        ani = animation.FuncAnimation(
            self.fig, update, frames=len(path), interval=self.timer_speed, blit=False
        )
        self.canvas.draw()