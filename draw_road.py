import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MCS import pressure
from model06 import satisfaction

# 粒子群优化参数
num_particles = 50  # 粒子数
max_iter = 100  # 最大迭代次数
dim = 2  # 目标函数的维度，这里只有两个变量（游客数量和税收）

# 粒子位置初始化
np.random.seed(42)
particles = np.random.rand(num_particles, dim) * 10  # 初始化粒子的位置，范围为[0, 10]，表示游客数量和税收
velocities = np.random.rand(num_particles, dim) * 2 - 1  # 粒子的速度


# 假设的四个目标函数
def objective(particle):
    # 游客数量和税收
    tourists = particle[0]
    tax = particle[1]

    expenses=2000
    # 目标1：收入（假设是游客数量与税收的函数）
    income = tourists * (tax+expenses)  # 收入 = 游客数量 * 税收

    # 目标2：生态环境压力（假设与游客数量有关，税收可能有抑制作用）
    #pressure = tourists ** 2 / (tax + 1)  # 生态压力假设与游客数量的平方成正比，税收对其有抑制作用
    pressure=tourists*1.2-tax*0.2

    # 目标3：居民满意度（假设税收高低会影响居民满意度）
    #satisfaction = 0.65+0.217 * tourists/(1+tourists) - 4.387 * tax  # 目标3：居民满意度，假设与游客数量正相关，税收负相关
    satisfaction=tourists*0.7-tax*0.2

    # 目标4：隐性成本（假设税收越高，隐性成本越高）
    cost = tax ** 2 - tourists  # 隐性成本假设与税收的平方相关，游客数量负相关

    return income, pressure, satisfaction, cost


# 更新粒子位置和速度
def update_particles(particles, velocities, pbest_positions, gbest_position, w=0.5, c1=1.5, c2=1.5):
    for i in range(num_particles):
        r1, r2 = np.random.rand(2)  # 随机数
        velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - particles[i]) + c2 * r2 * (
                    gbest_position - particles[i])
        particles[i] = particles[i] + velocities[i]
        # 确保粒子在范围内
        particles[i] = np.clip(particles[i], 0, 10)
    return particles, velocities


# 计算目标值并更新最优解
def calculate_fitness(particles):
    fitness_values = np.array([objective(p) for p in particles])
    return fitness_values


# 记录粒子历史位置
history_positions = []

# 初始化pbest（个体最优）和gbest（全局最优）
pbest_positions = particles.copy()
pbest_values = calculate_fitness(particles)
gbest_position = pbest_positions[np.argmin(np.sum(pbest_values, axis=1))]  # 根据四个目标的和来选择全局最优
gbest_value = pbest_values[np.argmin(np.sum(pbest_values, axis=1))]

# 主优化循环
for iteration in range(max_iter):
    fitness_values = calculate_fitness(particles)

    # 更新个体最优解
    for i in range(num_particles):
        if np.sum(fitness_values[i]) < np.sum(pbest_values[i]):
            pbest_positions[i] = particles[i]
            pbest_values[i] = fitness_values[i]

    # 更新全局最优解
    gbest_position = pbest_positions[np.argmin(np.sum(pbest_values, axis=1))]
    gbest_value = pbest_values[np.argmin(np.sum(pbest_values, axis=1))]

    # 更新粒子位置
    particles, velocities = update_particles(particles, velocities, pbest_positions, gbest_position)

    # 记录粒子位置历史
    history_positions.append(particles.copy())

# 绘制粒子位置和帕累托前沿
history_positions = np.array(history_positions)

# 获取帕累托前沿
pareto_front = pbest_values[np.argmin(np.sum(pbest_values, axis=1))]

# 使用平行坐标图展示多个目标
fig, ax = plt.subplots(figsize=(10, 6))

# 选择前几个粒子来绘制轨迹
for i in range(num_particles):
    ax.plot(history_positions[:, i, 0], history_positions[:, i, 1], marker='o', markersize=2,
            label=f"Particle {i}" if i < 5 else "", alpha=0.7)

# 帕累托最优解
plt.scatter(pbest_values[:, 0], pbest_values[:, 1], color='red', label="Pareto Optimal Solutions", edgecolor='black',
            s=50)

# 设置图表属性
plt.title('PSO Optimization with Pareto Front (Tourists and Tax)')
plt.xlabel('Tourist Numbers')
plt.ylabel('Tax')
plt.legend()
plt.grid(True)
plt.show()
