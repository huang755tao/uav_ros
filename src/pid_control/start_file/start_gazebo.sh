#!/bin/bash

# 定义路径变量（根据实际情况修改）
PX4_DIR="$HOME/src/PX4-Autopilot"
WORKSPACE_DIR="$HOME/src/px4_ws"
RVIZ_CONFIG="$WORKSPACE_DIR/src/pid_control/config/default_gazebo.rviz"
PYTHON_SCRIPT="$WORKSPACE_DIR/src/pid_control/scripts/odom_to_path_converter.py"

# 清理已有 ROS/Gazebo 进程（初始清理）
cleanup() {
    echo -e "\n清理所有进程..."
    pkill -f -9 "PX4 SITL|MAVROS Debug|Control Node|RVIZ|odom_to_path_converter|gzserver|gzclient|px4_sitl|roslaunch|rosrun"
    exit 0
}

# 关闭所有ros节点
kill_ros_nodes() {
    echo "Killing all ROS nodes..."
    rosnode list | xargs -r rosnode kill
    if [ $? -eq 0 ]; then
        echo "All ROS nodes have been killed successfully."
    else
        echo "Failed to kill some ROS nodes."
    fi
}
echo "Killing all ROS nodes..."
rosnode list | xargs -r rosnode kill
sleep 1

# 注册信号处理函数（捕获 Ctrl+C）
trap 'cleanup' INT

# 1. 启动 PX4 Gazebo 仿真环境（独立终端，标题唯一）
gnome-terminal --title="PX4 SITL" -- bash -c "cd $PX4_DIR && make px4_sitl gazebo; exec bash" &
px4_pid=$!

# 2. 启动 MAVROS 连接（独立终端，标题唯一）
gnome-terminal --title="MAVROS Debug" -- bash -c "source $WORKSPACE_DIR/devel/setup.bash && roslaunch mavros px4.launch fcu_url:='udp://:14540@127.0.0.1:14557'; exec bash" &
mavros_term_pid=$!

# 等待 Gazebo 启动（最多 20 秒）
echo "等待 Gazebo 启动..."
MAX_WAIT=20
while [ $MAX_WAIT -gt 0 ]; do
    if pgrep -x gzserver >/dev/null; then
        echo "Gazebo 启动成功"
        break
    fi
    sleep 1
    ((MAX_WAIT--))
done

if [ $MAX_WAIT -eq 0 ]; then
    echo "错误：Gazebo 未正常启动！"
    cleanup  # 调用清理函数关闭所有进程
fi

# 3. 启动控制节点（独立终端，标题唯一）
# gnome-terminal --title="Control Node" -- bash -c "source $WORKSPACE_DIR/devel/setup.bash && rosrun pid_control main.py; exec bash" &
# control_pid=$!

# # 4. 启动 odom_to_path_converter 节点（后台运行，记录 PID）
# source $WORKSPACE_DIR/devel/setup.bash
# rosrun pid_control $(basename $PYTHON_SCRIPT) &
# odom_pid=$!

# 5. 启动 RVIZ 可视化（独立终端，标题唯一）
gnome-terminal --title="RVIZ" -- bash -c "source /opt/ros/noetic/setup.bash && rosrun rviz rviz -d $RVIZ_CONFIG; exec bash" &
rviz_term_pid=$!

echo "所有组件已启动！按 Ctrl+C 关闭所有终端和进程..."

# 保持主终端运行，等待信号（Ctrl+C）
while true; do
    sleep 0.001
done

