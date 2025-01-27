##final


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 动态设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 假设的支出项对四个方面的影响力（带小数）
data = np.array([
    [8.2, 6.5, 5.8, 4.0],  # 生态环境维护投资
    [7.5, 5.9, 4.5, 3.5],  # 基础设施建设投资
    [6.0, 4.2, 6.3, 4.7],  # 其他旅游活动开发投资
    [5.2, 3.1, 5.4, 3.0],  # 饮用水增设设备投资
    [4.4, 3.8, 7.0, 4.6],  # 小众景点宣传推广投资
    [3.5, 6.0, 7.5, 6.0]   # 社会治安维护建设
])

# 定义四个方面和支出项
x_labels = ['旅游业总体收入', '生态环境压力', '当地居民满意度', '旅游业其他隐形成本']
y_labels = ['生态环境维护投资', '基础设施建设投资', '其他旅游活动开发投资',
            '饮用水增设设备投资', '小众景点宣传推广投资', '社会治安维护建设']

# 设置热力图的绘制
plt.figure(figsize=(10, 7), dpi=150)  # 调整图形尺寸和分辨率

# 绘制热力图
sns.heatmap(data, annot=True, cmap="Blues", xticklabels=x_labels, yticklabels=y_labels,
            cbar=True, annot_kws={'size': 14}, linewidths=0.5, linecolor='white', fmt='.1f')

# 设置标题和标签，增加字体大小
plt.title('旅游业额外收入支出项对四个方面的影响力热力图', fontsize=18, fontweight='bold')
plt.xlabel('')
plt.ylabel('支出计划', fontsize=14)  # 去掉y轴的标签

# 调整x轴标签的字体大小和方向
plt.xticks(fontsize=12, rotation=0)  # 水平显示x轴标签

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()
