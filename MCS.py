import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei' 之类
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 定义输入变量的概率分布
def generate_samples():
    # 基础设施投资，正态分布，均值100，标准差20
    investment_infrastructure = np.random.normal(100, 20, 1000)

    # 游客数量，均匀分布，范围[50000, 100000]
    tourist_number = np.random.uniform(50000, 100000, 1000)

    # 环境影响系数，正态分布，均值1.2，标准差0.5
    environmental_impact = np.random.normal(1.2, 0.5, 1000)

    return investment_infrastructure, tourist_number, environmental_impact


# 2. 计算模型输出：这里我们计算经济收入和环境压力
def calculate_outputs(investment_infrastructure, tourist_number, environmental_impact):
    # 假设经济收入 = 游客数量 * 价格（这里简化为100） + 投资
    revenue = tourist_number * 100 + investment_infrastructure * 0.5

    # 假设环境压力 = 游客数量 * 环境影响系数
    pressure = tourist_number * environmental_impact

    return revenue, pressure


# 3. 执行蒙特卡洛模拟
investment_infrastructure, tourist_number, environmental_impact = generate_samples()
revenue, pressure = calculate_outputs(investment_infrastructure, tourist_number, environmental_impact)

# 4. 敏感性分析：绘制收入和环境压力的敏感性图
# 相关性分析
correlation_revenue_investment = np.corrcoef(investment_infrastructure, revenue)[0, 1]
correlation_revenue_tourist = np.corrcoef(tourist_number, revenue)[0, 1]
correlation_pressure_impact = np.corrcoef(environmental_impact, pressure)[0, 1]

print(f"经济收入与基础设施投资的相关性: {correlation_revenue_investment}")
print(f"经济收入与游客数量的相关性: {correlation_revenue_tourist}")
print(f"环境压力与环境影响系数的相关性: {correlation_pressure_impact}")

# 5. 绘制敏感性图：收入与基础设施投资的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x=investment_infrastructure, y=revenue)
plt.title("经济收入与基础设施投资的关系")
plt.xlabel("基础设施投资")
plt.ylabel("经济收入")
plt.show()

# 绘制环境压力与环境影响系数的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x=environmental_impact, y=pressure)
plt.title("环境压力与环境影响系数的关系")
plt.xlabel("环境影响系数")
plt.ylabel("环境压力")
plt.show()

# 6. 敏感性排序图
sensitivity_scores = {
    '基础设施投资': correlation_revenue_investment,
    '游客数量': correlation_revenue_tourist,
    '环境影响系数': correlation_pressure_impact
}

# 根据相关性绘制敏感性排序图
sorted_sensitivity = sorted(sensitivity_scores.items(), key=lambda x: abs(x[1]), reverse=True)
variables = [x[0] for x in sorted_sensitivity]
scores = [x[1] for x in sorted_sensitivity]

plt.bar(variables, scores)
plt.title("各变量对模型输出的敏感性")
plt.xlabel("输入变量")
plt.ylabel("相关性")
plt.show()
