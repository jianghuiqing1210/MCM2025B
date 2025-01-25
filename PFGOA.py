import numpy as np


# 初始化鸽子种群
def initialize_population(pop_size, dim):
    return np.random.uniform(low=0, high=1, size=(pop_size, dim))


# 目标函数：经济收入 (简化版)
def objective_1(x):
    return np.sum(x)  # 这里假设游客数量 * 单价的线性组合


# 目标函数：生态环境压力 (简化版)
def objective_2(x):
    return np.sum(x ** 2)  # 假设环境压力和游客数量的平方成正比


# 目标函数：隐性成本 (简化版)
def objective_3(x):
    return np.sum(x * 0.5)  # 假设隐性成本和游客数量的半比例关系


# 目标函数：居民满意度 (简化版)
def objective_4(x):
    return np.sum(np.abs(x - 0.5))  # 假设满意度与游客数量的差异相关


# 计算适应度
def evaluate_fitness(x):
    f1 = objective_1(x)  # 经济收入
    f2 = objective_2(x)  # 生态环境压力
    f3 = objective_3(x)  # 隐性成本
    f4 = objective_4(x)  # 居民满意度
    return np.array([f1, f2, f3, f4])


# Pareto支配关系
def pareto_dominates(p, q):
    return all(p <= q) and any(p < q)


# PFGOA中的位置更新函数
def update_position(pigeon, global_best_position, alpha=0.5, beta=0.5):
    # 随机选择一个全局最优位置进行更新
    return pigeon + alpha * (pigeon - pigeon) + beta * (global_best_position - pigeon)


# PFGOA主函数
def pfgoa(pop_size, dim, max_iter):
    # 初始化鸽子种群
    population = initialize_population(pop_size, dim)
    personal_best = np.copy(population)
    personal_best_fitness = np.array([evaluate_fitness(p) for p in population])

    # 全局最优解
    global_best_position = population[np.argmin(personal_best_fitness[:, 0])]  # 选择经济收入最优解
    global_best_fitness = personal_best_fitness[np.argmin(personal_best_fitness[:, 0])]

    # 主迭代过程
    for iteration in range(max_iter):
        for i in range(pop_size):
            # 计算适应度
            fitness = evaluate_fitness(population[i])

            # 更新个体最优解
            if np.all(fitness <= personal_best_fitness[i]):
                personal_best[i] = population[i]
                personal_best_fitness[i] = fitness

            # 更新全局最优解
            if np.all(fitness <= global_best_fitness):
                global_best_position = population[i]
                global_best_fitness = fitness

        # 更新鸽子位置
        for i in range(pop_size):
            population[i] = update_position(population[i], global_best_position)

        # 输出当前Pareto前沿
        print(f"Iteration {iteration + 1}, Global Best: {global_best_fitness}")

    return global_best_position, global_best_fitness


# 参数设置
pop_size = 50  # 种群大小
dim = 10  # 每个鸽子位置的维度（决策变量个数）
max_iter = 100  # 最大迭代次数

# 执行PFGOA优化
best_position, best_fitness = pfgoa(pop_size, dim, max_iter)

# 输出最终结果
print("\nFinal Pareto Optimal Solution:")
print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")
