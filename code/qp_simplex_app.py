import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Целевая функция: f(x1, x2) = 2x1^2 + 3x2^2 + 4x1x2 - 6x1 - 3x2
def quadratic_func(x, y):
    return 2 * x**2 + 3 * y**2 + 4 * x * y - 6 * x - 3 * y

# Ограничения
def satisfies_constraints(x, y):
    return x + y <= 1 and 2 * x + 3 * y <= 4 and x >= 0 and y >= 0

def project(x, y):
    x, y = max(0, x), max(0, y)
    if x + y > 1:
        total = x + y
        x, y = x / total, y / total
    if 2 * x + 3 * y > 4:
        scale = 4 / (2 * x + 3 * y)
        x, y = x * scale, y * scale
    return x, y

# Симплекс-метод (Nelder-Mead) с учётом ограничений
def constrained_nelder_mead(func, x0, y0, step_size=0.1, max_iter=100, tol=1e-6):
    simplex = np.array([
        [x0, y0],
        [x0 + step_size, y0],
        [x0, y0 + step_size]
    ])
    simplex = np.array([project(x, y) for x, y in simplex])
    values = [func(x, y) for x, y in simplex]
    history = []

    for _ in range(max_iter):
        indices = np.argsort(values)
        simplex = simplex[indices]
        values = [values[i] for i in indices]
        best = simplex[0]
        worst = simplex[-1]
        second = simplex[-2]
        history.append((best[0], best[1], func(best[0], best[1])))

        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - worst)
        reflected = project(*reflected)
        f_reflected = func(*reflected)

        if f_reflected < values[0]:
            expanded = centroid + 2 * (centroid - worst)
            expanded = project(*expanded)
            f_expanded = func(*expanded)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                values[-1] = f_reflected
        elif f_reflected < values[-2]:
            simplex[-1] = reflected
            values[-1] = f_reflected
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            contracted = project(*contracted)
            f_contracted = func(*contracted)
            if f_contracted < values[-1]:
                simplex[-1] = contracted
                values[-1] = f_contracted
            else:
                simplex[1:] = best + 0.5 * (simplex[1:] - best)
                simplex[1:] = [project(x, y) for x, y in simplex[1:]]
                values = [func(x, y) for x, y in simplex]

        if np.std(values) < tol:
            break

    return np.array(history)

# GUI часть
class QPSimplexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Оптимизация")

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_panel = tk.Frame(main_frame, width=200)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.entries = {}
        for label, default in [
            ("Initial X1:", "0.5"),
            ("Initial X2:", "0.5"),
            ("Step Size:", "0.1"),
            ("Max Iterations:", "100"),
            ("Animation Speed:", "100"),
        ]:
            ttk.Label(control_panel, text=label).pack()
            entry = ttk.Entry(control_panel)
            entry.pack()
            entry.insert(0, default)
            self.entries[label] = entry

        self.start_button = ttk.Button(control_panel, text="Start", command=self.start)
        self.start_button.pack(pady=10)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def start(self):
        x0 = float(self.entries["Initial X1:"].get())
        y0 = float(self.entries["Initial X2:"].get())
        step = float(self.entries["Step Size:"].get())
        iters = int(self.entries["Max Iterations:"].get())
        speed = int(self.entries["Animation Speed:"].get())

        path = constrained_nelder_mead(quadratic_func, x0, y0, step, iters)

        for i, (x, y, z) in enumerate(path):
            print(f"{i}: (x: {x:.4f}, y: {y:.4f}, Значение = {z:.4f})")

        self.ax.clear()
        x_vals = np.linspace(0, 1.2, 100)
        y_vals = np.linspace(0, 1.2, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = quadratic_func(X, Y)
        self.ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.7)
        self.ax.set_title("Оптимизация")

        (scat,) = self.ax.plot([], [], [], "ko", markersize=5)
        (red_dot,) = self.ax.plot([], [], [], "ro", markersize=8)

        def update(frame):
            scatter_data = path[:frame+1]
            scat.set_data(scatter_data[:, 0], scatter_data[:, 1])
            scat.set_3d_properties(scatter_data[:, 2])
            red_dot.set_data([path[frame, 0]], [path[frame, 1]])
            red_dot.set_3d_properties([path[frame, 2]])
            return scat, red_dot

        ani = animation.FuncAnimation(self.fig, update, frames=len(path), interval=speed, blit=False)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = QPSimplexApp(root)
    root.mainloop()