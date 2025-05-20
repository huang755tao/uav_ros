import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具包


# 获取当前脚本的绝对路径（关键：明确数据目录为脚本所在目录）
data_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
# data_dir = os.path.join(script_dir, 'data')  # 假设数据文件在脚本同级的"data"文件夹（若数据和脚本同目录，可直接用script_dir）
os.makedirs(data_dir, exist_ok=True)  # 创建目录（可选，确保目录存在）

print(f"Data directory: {data_dir}")  # 打印数据目录，便于调试

# 查找所有符合条件的CSV文件（包含完整路径）
drone_files = [
    os.path.join(data_dir, f)  # 拼接完整路径
    for f in os.listdir(data_dir)
    if f.endswith('_uav_data.csv')
]
reference_files = [
    os.path.join(data_dir, f)  # 拼接完整路径
    for f in os.listdir(data_dir)
    if f.endswith('_ref_data.csv')
]

print("Drone files:", [os.path.basename(f) for f in drone_files])  # 打印文件名（调试用）
print("Reference files:", [os.path.basename(f) for f in reference_files])  # 打印文件名（调试用）

drone_files.sort()
reference_files.sort()

if not drone_files or not reference_files:
    print("Error: No drone data or reference trajectory files found.")
else:
    drone_filename = drone_files[-1]  # 最新的无人机数据文件（含完整路径）
    reference_filename = reference_files[-1]  # 最新的参考轨迹文件（含完整路径）

    try:
        # 读取数据（使用完整路径）
        drone_data = pd.read_csv(drone_filename)
        reference_data = pd.read_csv(reference_filename)
    except FileNotFoundError:
        print(f"错误：文件未找到\n无人机数据文件: {drone_filename}\n参考轨迹文件: {reference_filename}")
        exit(1)

    # 提取数据并转换为 numpy 数组
    drone_time = drone_data['timestamp'].to_numpy()
    drone_x = drone_data['x'].to_numpy()
    drone_y = drone_data['y'].to_numpy()
    drone_z = drone_data['z'].to_numpy()

    reference_time = reference_data['timestamp'].to_numpy()
    reference_x = reference_data['ref_x'].to_numpy()
    reference_y = reference_data['ref_y'].to_numpy()
    reference_z = reference_data['ref_z'].to_numpy()

    # 计算误差
    error_x = np.abs(drone_x - reference_x)
    error_y = np.abs(drone_y - reference_y)
    error_z = np.abs(drone_z - reference_z)

    # 绘制轨迹图
    fig = plt.figure(figsize=(15, 10))

    # 2D 轨迹图
    ax1 = fig.add_subplot(231)
    ax1.plot(reference_x, reference_y, label='Reference Trajectory', color='blue', linestyle='--')  # 添加参考轨迹
    ax1.plot(drone_x, drone_y, label='Drone Trajectory', color='red')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()

    # 3D 轨迹图
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot(reference_x, reference_y, reference_z, label='Reference Trajectory', color='blue', linestyle='--')  # 添加参考轨迹
    ax2.plot(drone_x, drone_y, drone_z, label='Drone Trajectory', color='red')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Z Position (m)')
    ax2.set_title('3D Trajectory Comparison')
    ax2.set_xlim(0,1.5)
    ax2.set_ylim(0,1.5)
    ax2.legend()

    # 误差曲线（添加时间对齐处理，假设时间戳一致，否则需插值）
    ax3 = fig.add_subplot(233)
    ax3.plot(drone_time, error_x, label='X Error', color='red')
    ax3.plot(drone_time, error_y, label='Y Error', color='green')
    ax3.plot(drone_time, error_z, label='Z Error', color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Error Curves')
    ax3.legend()

    # 位置曲线（添加时间对齐处理，假设时间戳一致，否则需插值）
    ax4 = fig.add_subplot(234)
    ax4.plot(drone_time, drone_x, label='X', color='red')
    ax4.plot(drone_time, reference_x, label='X REF', color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('x Curves')
    ax4.legend()

    ax5 = fig.add_subplot(235)
    ax5.plot(drone_time, drone_y, label='y', color='red')
    ax5.plot(drone_time, reference_y, label='Y REF', color='green')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('y Curves')
    ax5.legend() 

    ax6 = fig.add_subplot(236)
    ax6.plot(drone_time, drone_z, label='z', color='red')
    ax6.plot(drone_time, reference_z, label='Z REF', color='green')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title('z Curves')
    ax6.legend()

    plt.tight_layout()
    plt.show()
