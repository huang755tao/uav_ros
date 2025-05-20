#!/bin/bash

# 定义路径变量（根据实际情况修改）
PX4_DIR="$HOME/src/PX4-Autopilot"
WORKSPACE_DIR="$HOME/src/px4_ws"
RVIZ_CONFIG="$WORKSPACE_DIR/src/pid_control/rviz_config/drone_path.rviz"
PYTHON_SCRIPT="$WORKSPACE_DIR/src/pid_control/scripts/odom_to_path_converter.py"

# 清理已有 ROS/Gazebo 进程
echo "清理已有 ROS/Gazebo 进程..."
pkill -f -9 "roscore|rosout|gzserver|gzclient|px4_sitl"
sleep 5

# 1. 启动 PX4 Gazebo 仿真环境
gnome-terminal --title="PX4 SITL" -- bash -c "cd $PX4_DIR && make px4_sitl gazebo; exec bash"
sleep 5

# 2. 启动 MAVROS 连接
gnome-terminal --title="MAVROS" -- bash -c "roslaunch mavros px4.launch fcu_url:='udp://:14540@127.0.0.1:14557'; exec bash"


# 等待 PX4 初始化完成并检查 Gazebo 启动状态
echo "等待 Gazebo 启动..."
MAX_WAIT=30
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
    exit 1
fi


# 等待一段时间确保 MAVROS 有足够时间启动
sleep 5

# 3. 启动控制节点
# gnome-terminal --title="Control Node" -- bash -c "source $WORKSPACE_DIR/devel/setup.bash && rosrun pid_control circle_tracking.py; exec bash"

# 4. 启动 odom_to_path_converter 节点
# gnome-terminal --title="Odom to Path Converter" -- bash -c "source $WORKSPACE_DIR/devel/setup.bash && rosrun pid_control $(basename $PYTHON_SCRIPT); exec bash"
# sleep 2

# 5. 启动 RVIZ 可视化
# gnome-terminal --title="RVIZ" -- bash -c "source /opt/ros/noetic/setup.bash && rosrun rviz rviz; exec bash"

echo "所有组件已启动！"

