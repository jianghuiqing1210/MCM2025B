import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 目标函数 - 旅游业经济收入
def tourism_revenue(position):
    P0 = position[0]  # 每日游客人均花费
    pt = position[1]  # 每半年游客的季节性花费
    N_summer = position[2]  # 夏季游客数量
    N_winter = position[3]  # 冬季游客数量
    E = 1.2  # 季节性调整系数
    return (P0 + pt) * (N_summer + N_winter) * E

# 目标函数 - 生态环境压力
def environment_pressure(position):
    N_summer = position[2]  # 夏季游客数量
    N_winter = position[3]  # 冬季游客数量
    E = 1.2  # 季节性调整系数
    G = 0.4 * (1 + 0.1 * (N_summer + N_winter))  # 生态环境脆弱度（动态调整）
    total_visitors = N_summer + N_winter
    return G * total_visitors * E  # 非线性关系

# 目标函数 - 居民满意度
def calculate_satisfaction(position, S_prime=0.65, alpha=0.5, beta=0.1):
    N_total = position[2] + position[3]  # 总游客数量
    S = S_prime + alpha * N_total / (1 + N_total) - beta * (N_total ** 2) / (1 + N_total ** 2)
    return np.clip(S, 0, 1)  # 限制满意度在 [0, 1] 范围内

# 整合目标函数
def calculate_fitness(position, weights=[0.5, -0.3, 0.2]):
    I = tourism_revenue(position)  # 旅游业经济收入
    P = environment_pressure(position)  # 生态环境压力
    S = calculate_satisfaction(position)  # 居民满意度

    # 归一化处理（动态范围）
    I_norm = (I - I_min) / (I_max - I_min)
    P_norm = (P - P_min) / (P_max - P_min)
    S_norm = (S - S_min) / (S_max - S_min)

    # 加权求和
    fitness = weights[0] * I_norm + weights[1] * P_norm + weights[2] * S_norm
    return fitness, I, P, S

# 粒子群优化
num_particles = 100  # 增加粒子数量
num_variables = 4
max_iter = 300  # 增加迭代次数

# 变量范围（基于实际数据）
x_min = np.array([10, 5, 3, 2])  # 最小值
x_max = np.array([100, 50, 30, 20])  # 最大值

# 目标函数范围（动态调整）
I_min, I_max = 100, 10000  # 旅游业经济收入范围
P_min, P_max = 10, 1000    # 生态环境压力范围
S_min, S_max = 0, 1        # 居民满意度范围

# 初始化粒子群
positions = np.random.uniform(x_min, x_max, (num_particles, num_variables))
velocities = np.random.uniform(-1, 1, (num_particles, num_variables))

# PSO参数（动态调整）
w = 0.9  # 增加惯性权重
c1 = 2.0  # 增加个体学习因子
c2 = 2.0  # 增加全局学习因子
v_max = 2.0  # 最大速度限制

# 初始化个体最优解和全局最优解
pbest = positions.copy()
pbest_fitness = np.array([calculate_fitness(pos) for pos in positions])
gbest = positions[np.argmax(pbest_fitness[:, 0])]  # 选择最大适应度的全局最优解
gbest_fitness = np.max(pbest_fitness[:, 0])

# 用于记录收敛图数据
global_best_fitness_values = []

# 粒子群优化迭代
for iter in range(max_iter):
    for i in range(num_particles):
        # 更新速度
        r1, r2 = np.random.rand(2)
        velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - positions[i]) + c2 * r2 * (gbest - positions[i])
        velocities[i] = np.clip(velocities[i], -v_max, v_max)  # 限制速度范围

        # 更新位置
        positions[i] = positions[i] + velocities[i]
        positions[i] = np.clip(positions[i], x_min, x_max)  # 限制位置范围

        # 计算新适应度
        fitness_values = calculate_fitness(positions[i])
        fitness = fitness_values[0]

        # 更新个体最优解
        if fitness > pbest_fitness[i, 0]:
            pbest[i] = positions[i]
            pbest_fitness[i] = fitness_values

        # 更新全局最优解
        if fitness > gbest_fitness:
            gbest = positions[i]
            gbest_fitness = fitness

    # 动态调整惯性权重
    w = 0.9 - 0.5 * (iter / max_iter)  # 线性递减

    # 记录每次迭代的全局最优适应度
    global_best_fitness_values.append(gbest_fitness)
    print(f"Iteration {iter + 1}, Global Best Fitness: {gbest_fitness}")

# 输出最优解
print("\n最优解：", gbest)
print("全局最优适应度：", gbest_fitness)

# 分析最优解的目标函数值
I, P, S = tourism_revenue(gbest), environment_pressure(gbest), calculate_satisfaction(gbest)
print("\n最优解的目标函数值：")
print("旅游业经济收入 (I):", I)
print("生态环境压力 (P):", P)
print("居民满意度 (S):", S)

# 敏感性分析
def sensitivity_analysis(gbest, delta=0.1):
    print("\n敏感性分析：")
    for i in range(len(gbest)):
        perturbed_solution = gbest.copy()
        perturbed_solution[i] += delta
        perturbed_fitness = calculate_fitness(perturbed_solution)[0]
        sensitivity = abs(perturbed_fitness - gbest_fitness) / delta
        print(f"变量 {i} 的敏感性: {sensitivity}")

sensitivity_analysis(gbest)

# 可视化
# 1. 收敛图
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), global_best_fitness_values, marker='o', color='b', label='Global Best Fitness')
plt.xlabel("Iterations")
plt.ylabel("Global Best Fitness")
plt.title("Convergence of PSO")
plt.grid(True)
plt.legend()
plt.show()

# 2. 粒子群分布图
def plot_particle_distribution(positions, gbest):
    plt.figure(figsize=(10, 6))
    # 绘制所有粒子的位置
    plt.scatter(positions[:, 0], positions[:, 1], color='b', label='Particles', alpha=0.5)
    # 绘制全局最优解
    plt.scatter(gbest[0], gbest[1], color='r', label='Best Solution')
    # 动态调整范围
    plt.xlim([np.min(positions[:, 0]) - 1, np.max(positions[:, 0]) + 1])
    plt.ylim([np.min(positions[:, 1]) - 1, np.max(positions[:, 1]) + 1])
    plt.xlabel("Tourism Revenue (P0 + pt)")
    plt.ylabel("Seasonal Spending (pt)")
    plt.title("Particle Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_particle_distribution(positions, gbest)

# 3. 帕累托前沿图
def plot_pareto_front(positions):
    I_values = [calculate_fitness(pos)[1] for pos in positions]  # 旅游业经济收入
    P_values = [calculate_fitness(pos)[2] for pos in positions]  # 生态环境压力
    S_values = [calculate_fitness(pos)[3] for pos in positions]  # 居民满意度

    # 绘制帕累托前沿
    plt.figure(figsize=(10, 6))
    plt.scatter(I_values, P_values, c=S_values, cmap='viridis', label='Solutions', alpha=0.5)
    plt.colorbar(label='Satisfaction')
    plt.scatter(tourism_revenue(gbest), environment_pressure(gbest), color='r', label='Best Solution')
    plt.xlabel("Tourism Revenue (I)")
    plt.ylabel("Environment Pressure (P)")
    plt.title("Pareto Front: Tourism Revenue vs. Environment Pressure")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_pareto_front(positions)

# 4. 三维散点图
def plot_3d_scatter(positions, gbest):
    I_values = [calculate_fitness(pos)[1] for pos in positions]  # 旅游业经济收入
    P_values = [calculate_fitness(pos)[2] for pos in positions]  # 生态环境压力
    S_values = [calculate_fitness(pos)[3] for pos in positions]  # 居民满意度

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制所有粒子的位置
    ax.scatter(I_values, P_values, S_values, color='b', label='Particles', alpha=0.5)
    # 绘制全局最优解
    ax.scatter(tourism_revenue(gbest), environment_pressure(gbest), calculate_satisfaction(gbest), color='r', label='Best Solution')
    ax.set_xlabel("Tourism Revenue (I)")
    ax.set_ylabel("Environment Pressure (P)")
    ax.set_zlabel("Satisfaction (S)")
    ax.set_title("3D View of Solutions")
    plt.legend()
    plt.show()

plot_3d_scatter(positions, gbest)