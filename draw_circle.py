import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# 动态设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 活动名称和收入数据（单位：美元）
activities = ['冰川徒步', '观鲸', '飞行观光', '狗拉雪橇', '雨林徒步', '海上皮划艇', '其他小众活动']
income = [700, 400, 420, 150, 90, 60, 75]

# 假设不同因素的评分（0 到 1之间，1是最好）
market_demand = [0.9, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3]  # 市场需求
resource_availability = [0.8, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4]  # 资源可用性
environmental_impact = [0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.3]  # 环境影响

# 权重设置
weights = {'market_demand': 0.5, 'resource_availability': 0.3, 'environmental_impact': 0.2}

# 计算每个活动的加权潜力值
potential_values = np.array(market_demand) * weights['market_demand'] + \
                   np.array(resource_availability) * weights['resource_availability'] + \
                   np.array(environmental_impact) * weights['environmental_impact']

# 数据准备：将潜力值与收入进行组合
total_income = np.array(income)
total_potential_values = potential_values * total_income  # 潜力值与收入的乘积

# 按潜力值从大到小排序
sorted_indices = np.argsort(total_potential_values)[::-1]
sorted_activities = np.array(activities)[sorted_indices]
sorted_income = total_income[sorted_indices]
sorted_potential_values = total_potential_values[sorted_indices]
sorted_colors = plt.cm.summer(np.linspace(0.2, 0.6, len(sorted_activities)))

# 创建图形和坐标轴，设置图表尺寸为 (12, 8)
fig, ax = plt.subplots(figsize=(12, 8))

# 堆积柱状图（水平），每个柱子不同的低饱和绿色
bars = ax.barh(sorted_activities, sorted_potential_values, label='活动潜力值', color=sorted_colors)

# 在柱上添加数值标签，并增大字体
for i in range(len(sorted_activities)):
    ax.text(sorted_potential_values[i] + 10, i, f'{sorted_potential_values[i]:,.0f}', va='center', fontsize=14)

# 设置标题和标签
ax.set_title('不同旅游活动对收入的潜力值', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('潜力值', fontsize=16, fontweight='bold')
ax.set_ylabel('旅游活动', fontsize=16, fontweight='bold')

# 设置图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
