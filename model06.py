import math

import numpy as np
import matplotlib.pyplot as plt

# 灰色关联度分析
def gray_relation_degree(series, reference):
    """ 计算灰色关联度 """
    # 标准化输入数据
    series_normalized = (series - np.min(series)) / (np.max(series) - np.min(series))
    reference_normalized = (reference - np.min(reference)) / (np.max(reference) - np.min(reference))

    # 计算灰色关联度
    delta_0 = np.abs(series_normalized - reference_normalized)
    delta_0_min = np.min(delta_0)
    delta_0_max = np.max(delta_0)
    delta_1 = 0.5 * (delta_0_max - delta_0_min)
    gray_relation = (delta_0_min + delta_1) / (delta_0 + delta_1)
    return np.mean(gray_relation)


# 假设数据
incomes = np.array([9000000,1500000,5000000,800000])  # 不同支出下的经济收入
investments = np.array([700000, 120000, 90000, 111130])  # 各项支出（如基础设施投资）

# 计算灰色关联度
gray_relation = gray_relation_degree(investments, incomes)
print("灰色关联度:", gray_relation)

# 粒子群优化（PSO）
num_particles = 30
num_variables = 2  # 例如：门票、住宿等四个决策变量
max_iter = 100

# 粒子初始化范围
x_min = np.array([2.1, 0.6])  # 最小值
x_max = np.array([100, 20])  # 最大值

# 初始化粒子位置和速度
positions = np.random.uniform(x_min, x_max, (num_particles, num_variables))
velocities = np.random.uniform(-1, 1, (num_particles, num_variables))


# 目标函数
def calculate_fitness(position):
    # 假设位置变量分别表示：门票价格、住宿费用、夏季游客数量、冬季游客数量
    x = position[0]  # 游客数量
    y = position[1]  # 税收

    # 计算旅游业经济收入
    I = (y+2000)*x  # 季节性调整系数为 1.2

    # 计算生态环境压力
    G = 0.4  # 生态环境脆弱度
    P = (G *pow(x,1.05))-math.log(y)  # 环境压力与游客数量相关

    # 计算居民满意度

    S = 100*0.65 - 2.217 * x / (1 + x) + 0.387 *y
    S = np.clip(S, 0, 100)  # 限制满意度在 [0, 1] 范围内

    # 计算适应度（加权求和）
    fitness = 0.3 * I - 0.4 * P + 0.3 * S
    return fitness, I, P, S


# 更新粒子群
w = 0.5  # 惯性权重
c1 = 1.5  # 个体最优学习因子
c2 = 1.5  # 全局最优学习因子

# 初始化个体最优解和全局最优解
pbest = positions.copy()
pbest_fitness = np.array([calculate_fitness(pos) for pos in positions])
gbest = positions[np.argmax([fitness[0] for fitness in pbest_fitness])]  # 选择最大适应度的全局最优解
gbest_fitness = np.max([fitness[0] for fitness in pbest_fitness])

fitness_history = []

for iter in range(max_iter):
    for i in range(num_particles):
        # 更新速度
        r1, r2 = np.random.rand(2)
        velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - positions[i]) + c2 * r2 * (gbest - positions[i])

        # 更新位置
        positions[i] = positions[i] + velocities[i]

        # 保证粒子位置在范围内
        positions[i] = np.clip(positions[i], x_min, x_max)

        # 计算新适应度
        fitness_i = calculate_fitness(positions[i])

        # 更新个体最优解
        if fitness_i[0] > pbest_fitness[i][0]:
            pbest[i] = positions[i]
            pbest_fitness[i] = fitness_i

        # 更新全局最优解
        if fitness_i[0] > gbest_fitness:
            gbest = positions[i]
            gbest_fitness = fitness_i[0]

    fitness_history.append(gbest_fitness)  # 保存每一代的最优适应度

    print(f"Iteration {iter + 1}, Global Best Fitness: {gbest_fitness}")

# 输出全局最优解
print("最优解：", gbest)
print("全局最优适应度：", gbest_fitness)

# 分析最优解的目标函数值
I, P, S = calculate_fitness(gbest)[1:]
print("\n最优解的目标函数值：")
print("旅游业经济收入 (I):", I)
print("生态环境压力 (P):", P)
print("居民满意度 (S):", S)

# 绘制适应度变化图
plt.figure(figsize=(10, 6))
plt.plot(fitness_history)
plt.title("Global Best Fitness over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Global Best Fitness")
plt.show()


# 系统动力学模型
def system_dynamics(income, environment_pressure, satisfaction):
    # 旅游业增长与收入和环境压力相关
    tourism_growth = income * 0.1 - environment_pressure * 0.02  # 减少环境压力对收入增长的负面影响

    # 环境退化与游客数量和环境压力相关，减少环境压力对环境退化的影响
    environment_degradation = environment_pressure * 0.1 + income * 0.01  # 减少环境压力的影响，使环境压力增加较慢

    # 新的满意度计算
    #satisfaction_growth = income * 0.005  # 满意度随着收入的增加而增加
    # satisfaction_drop = environment_pressure * 0.01  # 满意度受到环境压力的负面影响

    new_satisfaction = satisfaction+income*0.1-environment_pressure*0.05

    # 限制满意度在[0, 100]之间
    #new_satisfaction = np.clip(new_satisfaction, 0, 100)

    # 更新状态
    new_income = income + tourism_growth
    new_environment_pressure = environment_pressure + environment_degradation

    # 限制变量范围
    new_income = max(new_income, 0)
    new_environment_pressure = max(new_environment_pressure, 0)

    return new_income, new_environment_pressure, new_satisfaction


# 假设初始值
income = 20000  # 初始收入
environment_pressure = 70  # 初始环境压力
satisfaction = 50  # 初始满意度

# 模拟迭代
iterations = 50
income_history = []
pressure_history = []
satisfaction_history = []

for i in range(iterations):
    income, environment_pressure, satisfaction = system_dynamics(income, environment_pressure, satisfaction)
    income_history.append(income)
    pressure_history.append(environment_pressure)
    satisfaction_history.append(satisfaction)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), income_history, label="Income")
plt.plot(range(iterations), pressure_history, label="Environmental Pressure")
plt.plot(range(iterations), satisfaction_history, label="Satisfaction")
plt.legend()
plt.title("System Dynamics over Time")
plt.xlabel("Iterations")
plt.ylabel("Values")
plt.show()
