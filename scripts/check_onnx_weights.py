import numpy as np
import onnx
from onnx import numpy_helper


def check_onnx_weights_range(model_path, thresh=1000):
    """
    加载ONNX模型，遍历所有初始化器（权重），并打印其名称、形状、数据类型以及权重的最小值和最大值。
    """
    print(f"正在加载模型: {model_path}")

    # 1. 加载ONNX模型
    try:
        model = onnx.load(model_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    graph = model.graph
    initializers = graph.initializer

    if not initializers:
        print("模型中没有找到初始化器（权重）。")
        return

    print(f"找到 {len(initializers)} 个初始化器（权重/偏置）。\n")
    print("=" * 80)
    print(f"{'名称':<40} | {'形状':<20} | {'DType':<8} | 最小值 | 最大值")
    print("=" * 80)

    # 2. 遍历初始化器（权重）
    for initializer in initializers:
        name = initializer.name

        if "/" in name:
            continue

        # 将TensorProto（ONNX格式的张量）转换为NumPy数组
        try:
            weight_array = numpy_helper.to_array(initializer)
        except Exception as e:
            print(f"无法将初始化器 {name} 转换为NumPy数组: {e}")
            continue

        shape = weight_array.shape
        dtype = weight_array.dtype

        # 3. 计算权值范围
        if weight_array.size > 0:
            min_val = np.min(weight_array)
            max_val = np.max(weight_array)
        else:
            min_val = "N/A"
            max_val = "N/A"

        # 4. 打印结果
        if abs(min_val) > thresh or abs(max_val) > thresh:
            print(f"{name:<80} | {str(shape):<20} | {str(dtype):<8} | {min_val:^6.4g} | {max_val:^6.4g}")

    print("=" * 80)


# 替换为您的ONNX模型路径
model_file = R"D:\Temp\best_model.sim.onnx"
# 假设您的模型文件名为 model_file
check_onnx_weights_range(model_file, thresh=10)
