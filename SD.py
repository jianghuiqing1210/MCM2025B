import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei' 之类
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 模型参数
alpha = 0.01  # 每个游客对环境的影响程度
beta = 0.05   # 社会压力的增长速率
gamma = 0.02  # 政策对游客数量的影响系数
delta = 0.1   # 环境恢复的速度
N_threshold = 10000  # 社会可接受的最大游客数量
P_max = 0.2  # 政策限制强度

# 初始条件
N0 = 5000    # 初始游客数量
G0 = 100     # 初始环境质量（比如冰川健康指数）
S0 = 0       # 初始社会压力（0表示没有压力）

# 时间参数
t = np.linspace(0, 50, 500)  # 50天的模拟，每天一步

# 定义系统的微分方程
def model(y, t, alpha, beta, gamma, delta, N_threshold, P_max):
    N, G, S = y  # 游客数量、环境质量、社会压力

    # 政策调整（假设政策对游客数量的影响是线性的）
    P_t = P_max * (S / N_threshold)  # 基于社会压力，调整政策强度

    # 微分方程
    dNdt = gamma * (N_threshold - N) - P_t * N  # 游客数量变化
    dGdt = alpha * N - delta * G  # 环境质量变化
    dSdt = beta * (N / N_threshold)  # 社会压力变化

    return [dNdt, dGdt, dSdt]

# 初始条件
y0 = [N0, G0, S0]

# 使用odeint求解微分方程
sol = odeint(model, y0, t, args=(alpha, beta, gamma, delta, N_threshold, P_max))

# 绘制结果
plt.figure(figsize=(12, 8))

# 游客数量
plt.subplot(3, 1, 1)
plt.plot(t, sol[:, 0], 'b-', label='游客数量 (N)')
plt.xlabel('时间')
plt.ylabel('游客数量')
plt.legend(loc='best')

# 环境质量
plt.subplot(3, 1, 2)
plt.plot(t, sol[:, 1], 'g-', label='环境质量 (G)')
plt.xlabel('时间')
plt.ylabel('环境质量')
plt.legend(loc='best')

# 社会压力
plt.subplot(3, 1, 3)
plt.plot(t, sol[:, 2], 'r-', label='社会压力 (S)')
plt.xlabel('时间')
plt.ylabel('社会压力')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
