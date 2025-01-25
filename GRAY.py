import numpy as np
import pandas as pd


# 1. 数据标准化
def normalize_data(data):
    """
    将数据进行标准化处理，使得每列数据的值在0到1之间
    :param data: 输入的原始数据，DataFrame形式
    :return: 标准化后的数据
    """
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data


# 2. 计算灰色关联度
def gray_relation(X, Y):
    """
    计算灰色关联度
    :param X: 目标序列（例如：经济收入目标）
    :param Y: 评价序列（例如：各项支出）
    :return: 灰色关联度
    """
    # 计算绝对差
    delta_0 = np.abs(X - Y)
    print("delta_0 (目标序列和评价序列的绝对差):", delta_0)  # 调试：打印差值

    # 计算最大差和最小差
    delta_1 = np.max(delta_0)  # 最大差
    delta_0_min = np.min(delta_0)  # 最小差

    print("最大差 delta_1:", delta_1)  # 调试：打印最大差
    print("最小差 delta_0_min:", delta_0_min)  # 调试：打印最小差

    # 避免除以零的情况，添加一个很小的常数
    epsilon = 1e-6
    gray_relation_degree = (delta_0_min + 0.5 * delta_1) / (delta_0 + 0.5 * delta_1 + epsilon)

    return gray_relation_degree


# 3. 主程序
def main():
    # 假设你们有一个DataFrame，其中包含不同支出项和目标的数据
    data = pd.DataFrame({
        '经济收入': [100, 150, 200, 250, 300],
        '基础设施投资': [20, 30, 40, 50, 60],
        '环境保护费用': [5, 10, 15, 20, 25],
        '社会治理成本': [10, 15, 20, 25, 30],
        '门票收入': [50, 75, 100, 125, 150],
        '住宿收入': [30, 45, 60, 75, 90],
    })

    # 2. 标准化数据
    normalized_data = normalize_data(data)

    # 输出标准化后的数据进行调试
    print("标准化后的数据：")
    print(normalized_data)

    # 3. 目标序列（假设目标为经济收入）
    target = normalized_data['经济收入'].values  # 选择目标为经济收入

    # 4. 计算每个支出项与目标的灰色关联度
    gray_relations = {}
    for column in normalized_data.columns[1:]:  # 排除目标列
        gray_relations[column] = gray_relation(target, normalized_data[column].values)

    # 输出关联度结果
    print("\n各支出项与经济收入的灰色关联度：")
    for key, value in gray_relations.items():
        print(f"{key}: {value}")

    # 5. 根据关联度进行优化（例如，选择关联度较大的项进行更多投资）
    # 解决排序时出现的错误
    sorted_relations = sorted(gray_relations.items(), key=lambda x: np.max(x[1]), reverse=True)  # 使用np.max来确保关联度是标量值

    print("\n根据关联度排序后的支出项：")
    for item in sorted_relations:
        print(f"{item[0]}: {item[1]}")


# 调用主程序
if __name__ == "__main__":
    main()
