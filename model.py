import numpy as np
import random


# 1. 定义目标函数
def economic_revenue(x, investment, gray_coeffs):
    # 经济收入：根据游客数量和设施投资
    revenue = sum(x) * gray_coeffs['tourist_price'] + investment['infrastructure'] * gray_coeffs[
        'infrastructure_impact']
    return revenue


def environmental_pressure(x, investment, gray_coeffs):
    # 环境压力：根据游客数量和环保投资
    pressure = sum(x) * gray_coeffs['environmental_impact'] - investment['environmental_protection'] * gray_coeffs[
        'eco_investment_impact']
    return pressure


def hidden_cost(x, investment, gray_coeffs):
    # 隐性成本：基础设施维护成本，社会治理成本
    cost = investment['infrastructure'] * gray_coeffs['infra_cost'] + investment['social_governance'] * gray_coeffs[
        'soc_cost']
    return cost


def resident_satisfaction(x, investment, gray_coeffs):
    # 居民满意度：依赖于收入、环境质量、交通压力
    satisfaction = gray_coeffs['income_weight'] * economic_revenue(x, investment, gray_coeffs) \
                   + gray_coeffs['environment_weight'] * (1 / (environmental_pressure(x, investment, gray_coeffs) + 1)) \
                   - gray_coeffs['traffic_weight'] * investment['infrastructure']
    return satisfaction


# 2. 定义目标函数列表
def objective_functions(x, investment, gray_coeffs):
    I = economic_revenue(x, investment, gray_coeffs)
    P = environmental_pressure(x, investment, gray_coeffs)
    C = hidden_cost(x, investment, gray_coeffs)
    S = resident_satisfaction(x, investment, gray_coeffs)
    return np.array([I, P, C, S])


# 3. PFGOA算法优化
def PFGOA(pop_size, max_iter, gray_coeffs, investment_plan):
    # 初始化鸽子位置（即决策变量）和适应度（目标函数值）
    population = np.random.rand(pop_size, 4)  # 假设4个决策变量
    best_positions = population.copy()
    best_fitness = np.inf * np.ones((pop_size, 4))  # 假设初始适应度为无穷大

    global_best_position = None
    global_best_fitness = np.inf

    for iter in range(max_iter):
        for i in range(pop_size):
            # 计算适应度（目标函数值）
            fitness = objective_functions(population[i], investment_plan, gray_coeffs)

            # 更新个体最优
            if np.all(fitness < best_fitness[i]):
                best_fitness[i] = fitness
                best_positions[i] = population[i]

            # 更新全局最优
            if np.all(fitness < global_best_fitness):
                global_best_fitness = fitness
                global_best_position = population[i]

        # 更新鸽子的位置
        for i in range(pop_size):
            # 简化的更新公式
            population[i] = best_positions[i] + np.random.rand() * (global_best_position - population[i])

        print(f"第 {iter} 轮：最优适应度 = {global_best_fitness}")

    return global_best_position, global_best_fitness


# 4. 灰色关联度反馈（示例）
def gray_relation_feedback(gray_coeffs, investment_plan, global_best_position):
    # 假设灰色关联度影响因子来自灰色评价模型
    # 对支出项进行动态调整（简单模拟）
    gray_relation = np.random.rand(4)  # 假设灰色关联度反馈是随机的（实际应该根据前面的模型计算）
    adjusted_investment = investment_plan.copy()
    adjusted_investment['infrastructure'] += gray_relation[0] * global_best_position[0]
    adjusted_investment['environmental_protection'] += gray_relation[1] * global_best_position[1]
    adjusted_investment['social_governance'] += gray_relation[2] * global_best_position[2]
    return adjusted_investment


# 5. 系统动力学反馈
def system_dynamics_feedback(investment_plan, adjusted_investment):
    # 动态调整：假设通过反馈调整支出项
    new_investment_plan = investment_plan.copy()
    for key in investment_plan:
        new_investment_plan[key] += 0.1 * (adjusted_investment[key] - investment_plan[key])
    return new_investment_plan


# 6. 循环过程
def feedback_optimization():
    # 初始投资计划（支出项）
    investment_plan = {
        'infrastructure': 100,
        'environmental_protection': 50,
        'social_governance': 30
    }

    # 灰色关联度系数（假设灰色关联度从灰色评价模型得到）
    gray_coeffs = {
        'tourist_price': 20,
        'environmental_impact': 1.5,
        'infrastructure_impact': 0.8,
        'eco_investment_impact': 0.3,
        'income_weight': 0.4,
        'environment_weight': 0.3,
        'traffic_weight': 0.3,
        'infra_cost': 0.5,
        'soc_cost': 0.2
    }

    # 优化过程：进行多次循环优化
    max_iter = 50
    pop_size = 10
    for cycle in range(10):  # 进行多次反馈循环
        print(f"第 {cycle} 轮 - 开始")

        # 使用PFGOA进行优化
        global_best_position, global_best_fitness = PFGOA(pop_size, max_iter, gray_coeffs, investment_plan)

        # 反馈机制：调整支出计划
        adjusted_investment = gray_relation_feedback(gray_coeffs, investment_plan, global_best_position)

        # 系统动力学反馈：根据调整后的支出重新规划投资
        investment_plan = system_dynamics_feedback(investment_plan, adjusted_investment)

        print(f"第 {cycle} 轮 - 优化后的投资计划: {investment_plan}")
        print(f"第 {cycle} 轮 - 最优适应度: {global_best_fitness}")

    return global_best_position, global_best_fitness


# 执行反馈优化
final_best_position, final_best_fitness = feedback_optimization()
print("最终最优解位置:", final_best_position)
print("最终最优适应度:", final_best_fitness)
