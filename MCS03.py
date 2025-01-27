import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 系统动力学模型
def system_dynamics(income, environment_pressure, satisfaction):
    tourism_growth = income * 0.05 - environment_pressure * 0.005  # 旅游业增长
    environment_degradation = environment_pressure * 0.03 + income * 0.003  # 环境退化
    satisfaction_growth = income * 0.001  # 满意度增长
    satisfaction_drop = environment_pressure * 0.005  # 满意度下降
    new_satisfaction = satisfaction + satisfaction_growth - satisfaction_drop
    new_satisfaction = np.clip(new_satisfaction, 0, 100)

    # 更新收入和环境压力
    new_income = income + tourism_growth
    new_environment_pressure = environment_pressure + environment_degradation

    return new_income, new_environment_pressure, new_satisfaction

# 蒙特卡洛模拟函数
def monte_carlo_simulation(num_simulations, income_range, pressure_range, satisfaction_range):
    income_results = []
    pressure_results = []
    satisfaction_results = []

    for _ in range(num_simulations):
        income = np.random.uniform(*income_range)  # 随机生成收入
        environment_pressure = np.random.uniform(*pressure_range)  # 随机生成环境压力
        satisfaction = np.random.uniform(*satisfaction_range)  # 随机生成满意度

        # 运行系统动力学模型
        income_result, pressure_result, satisfaction_result = system_dynamics(income, environment_pressure, satisfaction)

        # 保存结果
        income_results.append(income_result)
        pressure_results.append(pressure_result)
        satisfaction_results.append(satisfaction_result)

    return income_results, pressure_results, satisfaction_results

# 设置输入变量范围
income_range = (4000, 12000)
pressure_range = (50, 300)
satisfaction_range = (60, 90)

# 进行蒙特卡洛模拟
num_simulations = 1000
income_results, pressure_results, satisfaction_results = monte_carlo_simulation(num_simulations, income_range, pressure_range, satisfaction_range)

# 数据标准化
scaler = StandardScaler()
income_results_np = np.array(income_results).reshape(-1, 1)
pressure_results_np = np.array(pressure_results).reshape(-1, 1)
satisfaction_results_np = np.array(satisfaction_results).reshape(-1, 1)

income_scaled = scaler.fit_transform(income_results_np).reshape(-1)
pressure_scaled = scaler.fit_transform(pressure_results_np).reshape(-1)
satisfaction_scaled = scaler.fit_transform(satisfaction_results_np).reshape(-1)

# 创建分开输出的图形

# --- 1. 条形图：每个因素单独展示 ---
plt.figure(figsize=(12, 8))

# 绘制收入的条形图
plt.subplot(3, 1, 1)
plt.hist(income_scaled, bins=30, color='skyblue', edgecolor='black')
plt.title("Income Distribution")
plt.xlabel("Income (Standardized)")
plt.ylabel("Frequency")

# 绘制环境压力的条形图
plt.subplot(3, 1, 2)
plt.hist(pressure_scaled, bins=30, color='lightgreen', edgecolor='black')
plt.title("Environmental Pressure Distribution")
plt.xlabel("Environmental Pressure (Standardized)")
plt.ylabel("Frequency")

# 绘制满意度的条形图
plt.subplot(3, 1, 3)
plt.hist(satisfaction_scaled, bins=30, color='salmon', edgecolor='black')
plt.title("Satisfaction Distribution")
plt.xlabel("Satisfaction (Standardized)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# --- 2. 箱线图：将三个因素整合到一张图 ---
plt.figure(figsize=(12, 6))

# 整合所有数据
data_scaled = [income_scaled, pressure_scaled, satisfaction_scaled]

# 绘制箱线图
plt.boxplot(data_scaled, vert=False, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            whiskerprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'))

plt.title("Boxplot of Income, Environmental Pressure, and Satisfaction")
plt.xlabel("Value (Standardized)")
plt.yticks([1, 2, 3], ['Income', 'Environmental Pressure', 'Satisfaction'])
plt.show()

# 计算输出的统计量
income_mean = np.mean(income_results)
pressure_mean = np.mean(pressure_results)
satisfaction_mean = np.mean(satisfaction_results)

income_std = np.std(income_results)
pressure_std = np.std(pressure_results)
satisfaction_std = np.std(satisfaction_results)

print(f"Income: Mean = {income_mean:.2f}, Std = {income_std:.2f}")
print(f"Environmental Pressure: Mean = {pressure_mean:.2f}, Std = {pressure_std:.2f}")
print(f"Satisfaction: Mean = {satisfaction_mean:.2f}, Std = {satisfaction_std:.2f}")