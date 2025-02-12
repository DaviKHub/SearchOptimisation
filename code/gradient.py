import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Функция для оптимизации
def f(x1, x2):
    return x1**2 + 2*x2**2 + 2*x1*x2

# Градиент функции
def grad_f(x1, x2):
    df_dx1 = 2*x1 + 2*x2
    df_dx2 = 4*x2 + 2*x1
    return np.array([df_dx1, df_dx2])

# Алгоритм градиентного спуска
alpha = 0.1  # шаг спуска
max_iters = 50  # число итераций
x_history = []  # история точек

x = np.array([0.5, 1])  # начальная точка
for i in range(max_iters):
    x_history.append(x.copy())
    grad = grad_f(x[0], x[1])
    x = x - alpha * grad
x_history = np.array(x_history)

# Визуализация
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Создаем сетку для отображения функции
X1 = np.linspace(-2, 2, 100)
X2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(X1, X2)
Z = f(X1, X2)

ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
ax.set_title('Градиентный спуск')

# Анимация градиентного спуска
line, = ax.plot([], [], [], 'r-', marker='o', markersize=5)

def update(frame):
    line.set_data(x_history[:frame+1, 0], x_history[:frame+1, 1])
    line.set_3d_properties(f(x_history[:frame+1, 0], x_history[:frame+1, 1]))
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x_history), interval=200, blit=False)
plt.show()
