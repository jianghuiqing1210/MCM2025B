import numpy as np
import matplotlib.pyplot as plt

# 目标函数 - 旅游业经济收入
def tourism_revenue(position):
    P0 = position[0]  # 每日游客人均花费
    pt = position[1]  # 每半年游客的季节性花费
    N_summer = position[2]  # 夏季游客数量
    N_winter = position[3]  # 冬季游客数量
    E = 1.2  # 季节性调整系数（可以根据需要修改）

    return (P0 + pt) * (N_summer + N_winter) * E

# 目标函数 - 生态环境压力（脆弱度计算）
def environment_pressure(position):
    N_summer = position[2]  # 夏季游客数量
    N_winter = position[3]  # 冬季游客数量
    E = 1.2  # 季节性调整系数（可以根据需要修改）
    G = 0.4  # 生态环境脆弱度（根据前面的公式计算）

    total_visitors = N_summer + N_winter
    return G * total_visitors * E

# 目标函数 - 居民满意度
def calculate_satisfaction(position, S_prime=0.65, alpha=0.217, beta=4.387):
    N_total = position[2] + position[3]  # 总游客数量（夏季 + 冬季）
    return S_prime + alpha * N_total / (1 + N_total) - beta * N_total ** 2

# 整合目标函数
def calculate_fitness(position):
    I = tourism_revenue(position)  # 旅游业经济收入
    P = environment_pressure(position)  # 生态环境压力
    C = np.sum(position) * 0.2  # 隐性成本
    S = calculate_satisfaction(position)  # 居民满意度

    return I, P, C, S

# 粒子群优化
num_particles = 30
num_variables = 4
max_iter = 100

x_min = np.array([10, 5, 3, 2])  # 最小值
x_max = np.array([100, 50, 30, 20])  # 最大值

positions = np.random.uniform(x_min, x_max, (num_particles, num_variables))
velocities = np.random.uniform(-1, 1, (num_particles, num_variables))

# PSO参数
w = 0.5  # 惯性权重
c1 = 1.5  # 个体最优学习因子
c2 = 1.5  # 全局最优学习因子

# 初始化个体最优解和全局最优解
pbest = positions.copy()
pbest_fitness = np.array([calculate_fitness(pos) for pos in positions])
gbest = positions[np.argmin(pbest_fitness[:, 0])]  # 选择最大收入的全局最优解
gbest_fitness = np.max(pbest_fitness[:, 0])

# 用于记录收敛图数据
global_best_fitness_values = []  # 初始化全局最优适应度记录列表

# 粒子群优化迭代
for iter in range(max_iter):
    for i in range(num_particles):
        # 更新速度
        r1, r2 = np.random.rand(2)
        velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - positions[i]) + c2 * r2 * (gbest - positions[i])

        # 更新位置
        positions[i] = positions[i] + velocities[i]

        # 保证粒子位置在范围内
        positions[i] = np.clip(positions[i], x_min, x_max)

        # 计算新适应度（考虑三个目标函数）
        fitness_values = np.array([calculate_fitness(positions[i])])
        R = fitness_values[0, 0]  # 计算旅游业经济收入
        P = fitness_values[0, 1]  # 计算生态环境压力
        S = fitness_values[0, 3]  # 计算居民满意度

        # 计算适应度
        fitness = R - P + S

        # 更新个体最优解
        if fitness > pbest_fitness[i, 0]:
            pbest[i] = positions[i]
            pbest_fitness[i] = fitness_values

        # 更新全局最优解
        if fitness > gbest_fitness:
            gbest = positions[i]
            gbest_fitness = fitness

    # 记录每次迭代的全局最优适应度
    global_best_fitness_values.append(gbest_fitness)
    print(f"Iteration {iter + 1}, Global Best Fitness: {gbest_fitness}")

# 输出最优解
print("最优解：", gbest)
print("全局最优适应度：", gbest_fitness)

# 结果可视化
# 收敛图：展示全局最优适应度的变化
def plot_convergence(iterations, global_best_fitness_values):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, global_best_fitness_values, marker='o', color='b', label='Global Best Fitness')
    plt.title("Convergence of PSO (Global Best Fitness)")
    plt.xlabel("Iterations")
    plt.ylabel("Global Best Fitness")
    plt.grid(True)
    plt.legend()
    plt.show()

# 目标函数值分布图：展示每个粒子在目标函数上的表现
def plot_fitness_distribution(fitness_values):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 旅游业经济收入分布
    ax[0].hist(fitness_values[:, 0], bins=20, color='g', alpha=0.7)
    ax[0].set_title("Tourism Revenue Distribution")
    ax[0].set_xlabel("Tourism Revenue")
    ax[0].set_ylabel("Frequency")

    # 生态环境压力分布
    ax[1].hist(fitness_values[:, 1], bins=20, color='r', alpha=0.7)
    ax[1].set_title("Environment Pressure Distribution")
    ax[1].set_xlabel("Environment Pressure")
    ax[1].set_ylabel("Frequency")

    # 居民满意度分布
    ax[2].hist(fitness_values[:, 3], bins=20, color='b', alpha=0.7)
    ax[2].set_title("Satisfaction Distribution")
    ax[2].set_xlabel("Satisfaction")
    ax[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# 散点图：展示粒子群优化后最优解
def plot_best_solution(best_solution, x_min, x_max):
    # best_solution 为粒子群的全局最优解
    plt.figure(figsize=(8, 6))

    plt.scatter(best_solution[0], best_solution[1], color='r', label='Best Solution')
    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])

    plt.title("Best Solution (Tourism Revenue vs. Environment Pressure)")
    plt.xlabel("Tourism Revenue")
    plt.ylabel("Environment Pressure")
    plt.legend()
    plt.grid(True)
    plt.show()

# 可视化
iterations = np.arange(1, max_iter + 1)  # 假设有 max_iter 次迭代
fitness_values = np.random.random((num_particles, 4)) * 1000  # 假设的适应度值，替换为实际结果

# 调用可视化函数
plot_convergence(iterations, global_best_fitness_values)
plot_fitness_distribution(fitness_values)
plot_best_solution(gbest, x_min, x_max)
