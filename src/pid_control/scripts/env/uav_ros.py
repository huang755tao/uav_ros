import numpy as np
import rospy
from pathlib import Path as Filepath
from datetime import datetime
import pandas as pd

from utils import *
from env.agent_init_v2 import AgentInit


class UAV_ROS(AgentInit):
	"""
	UAV ROS Control Class for state acquisition, dynamics calculation, control command publishing, and data logging

	Attributes:
		m (float): UAV mass (kg)
		g (float): Gravitational acceleration (m/s²)
		kt (float): Translational damping coefficient
		dt (float): Control cycle (s)
		use_gazebo (bool): Whether to use Gazebo simulation
		throttle (float): Current total thrust (N)
		phi_d/theta_d/psi_d (float): Desired roll/pitch/yaw angles (rad)
		data_log (list): UAV state log
		ref_data_log (list): Reference trajectory log
	"""
	def __init__(self, m: float = 1.5, 
			  		   g: float = 9.8, 
					   kt: float = 1e-3, 
					   dt: float = 0.02, 
					   use_gazebo: bool = True, 
					   control_name: str = ""):
		super().__init__(dt=dt, use_gazebo=use_gazebo)

		self.m = m  # UAV mass
		self.g = g  # Gravitational acceleration
		self.kt = kt  # Translational damping coefficient

		self.x = 0.0
		self.y = 0.0
		self.z = 0.0

		self.vx = 0.0
		self.vy = 0.0
		self.vz = 0.0

		self.phi = 0.0
		self.theta = 0.0
		self.psi = 0.0

		self.p = 0.0
		self.q = 0.0
		self.r = 0.0

		self.dt = dt
		self.n = 0  # Number of time steps
		self.time = 0.0  # Current time

		self.is_record_ref = False
		self.is_record_obs = False

		self.ref_data_log = []
		self.obs_data_log = []
		
		self.control_name = control_name

		'''Control parameters'''
		self.throttle = self.m * self.g  # Throttle (total thrust)
		self.phi_d = 0.0
		self.theta_d = 0.0
		self.psi_d = 0.0  # Added: Desired yaw angle
		'''Control parameters'''

	# -------------------- State acquisition methods --------------------
	def uav_state_callback(self) -> np.ndarray:
		"""Get full UAV state"""
		return self.uav_states  # Assumed to be maintained by parent class AgentInit

	def uav_pos(self) -> np.ndarray:
		"""Get position [x, y, z]"""
		return self.uav_states[0:3]

	def uav_vel(self) -> np.ndarray:
		"""Get velocity [vx, vy, vz]"""
		return self.uav_states[3:6]

	def uav_att(self) -> np.ndarray:
		"""Get attitude [phi, theta, psi]"""
		return self.uav_states[6:9]

	def uav_pqr(self) -> np.ndarray:
		"""Get angular velocity [p, q, r]"""
		return self.uav_states[9:12]

	# -------------------- Dynamics & kinematics methods --------------------
	def T_pqr_2_dot_att(self) -> np.ndarray:
		"""
		Transformation matrix from angular velocities (pqr) to attitude rates (φ̇θ̇ψ̇)
		Returns:
			np.ndarray: 3x3 transformation matrix
		"""
		[self.phi, self.theta, self.psi] = self.uav_att()
		# Ensure attitudes are scalars (prevent array input to sin/cos)
		assert all(isinstance(angle, (int, float)) for angle in [self.phi, self.theta]), "Attitude angles must be scalars"
		return np.array([
			[1, np.sin(self.phi) * np.tan(self.theta), np.cos(self.phi) * np.tan(self.theta)],
			[0, np.cos(self.phi), -np.sin(self.phi)],
			[0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]
		])

	def uav_dot_att(self) -> np.ndarray:
		"""
		Calculate attitude rates (φ̇, θ̇, ψ̇)
		Returns:
			np.ndarray: Attitude rate vector (rad/s)
		"""
		return np.dot(self.T_pqr_2_dot_att(), self.uav_pqr())

	def A(self) -> np.ndarray:
		"""
		Calculate UAV acceleration (considering gravity and attitude transformation)
		Returns:
			np.ndarray: Acceleration vector [ax, ay, az] (m/s²)
		"""
		[self.phi, self.theta, self.psi] = self.uav_att()
		thrust_orientation = np.array([
			np.cos(self.phi) * np.cos(self.psi) * np.sin(self.theta) + np.sin(self.phi) * np.sin(self.psi),
			np.cos(self.phi) * np.sin(self.psi) * np.sin(self.theta) - np.sin(self.phi) * np.cos(self.psi),
			np.cos(self.phi) * np.cos(self.theta)
		])
		return (self.throttle / self.m) * thrust_orientation - np.array([0.0, 0.0, self.g])

	def u_to_angle_dir(self, uo: np.ndarray, is_idea=True):
		"""
		Convert desired acceleration to desired attitude and total thrust
		Args:
			uo: Desired acceleration [ax, ay, az] (m/s²)
		Returns:
			(phi_d, theta_d, thrust): Desired roll, pitch angles, and total thrust
		"""
		[self.phi, self.theta, self.psi] = self.uav_att()

		u1 = uo + np.array([0., 0., self.g])  # Compensate for gravity

		uf = self.m * np.linalg.norm(u1)         # Total thrust
		
		# Vertical acceleration cannot be less than -g (prevent negative thrust)
		if uo[2] + self.g <= 0:
			rospy.logwarn("Vertical acceleration cannot be less than -g, using current attitude")
			return (self.phi, self.theta, 0.0)
		
		# Total thrust cannot be zero (prevent division by zero)
		if uf == 0:
			rospy.logwarn("Total thrust is zero, using current attitude")
			return (self.phi, self.theta, 0.0)
		
		# Calculate desired roll angle (clamped to [-π/2, π/2])
		sin_phi_dir = np.clip(
			(self.m * (uo[0] * np.sin(self.psi) - uo[1] * np.cos(self.psi))) / uf,
			-1.0, 1.0
		)
		phi_d = np.arcsin(sin_phi_dir)
		
		# Calculate desired pitch angle
		tan_theta_dir = (uo[0] * np.cos(self.psi) + uo[1] * np.sin(self.psi)) / u1[2]
		theta_d = np.arctan(tan_theta_dir)
		# if not is_idea:
		# 	# u1 += np.array([3.5, 3.5, 2])
		# 	uf = uf * 0.7
		
		return (phi_d, theta_d, uf)

	# -------------------- Logging and shutdown handling --------------------
	def data_record(self, sim_t: float, ref_state: np.ndarray=np.zeros(9), obs_state: np.ndarray=np.zeros(3)) -> None:
		"""
		Log reference trajectory data (strict dimension check)
		Args:
			sim_t: Current simulation time (s)
			ref_state: Reference state [x, y, z, vx, vy, vz, ax, ay, az]
		"""
		self.data_log.append({
			"timestamp": sim_t,
			"x":self.uav_states[0], "y":self.uav_states[1], "z":self.uav_states[2],
			"vx":self.uav_states[3], "vy":self.uav_states[4], "vz":self.uav_states[5],
			"phi":self.uav_states[6], "theta":self.uav_states[7], "psi":self.uav_states[8],
			"p":self.uav_states[9], "q":self.uav_states[10], "r":self.uav_states[11]})

		if self.is_record_ref:
			if ref_state.ndim != 1 or ref_state.size != 9:
				raise ValueError(f"Reference state dimension error, expected (9, ), got {ref_state.shape}")
			
			self.ref_data_log.append({
				"timestamp": sim_t,
				"ref_x": ref_state[0], "ref_y": ref_state[1], "ref_z": ref_state[2],
				"ref_vx": ref_state[3], "ref_vy": ref_state[4], "ref_vz": ref_state[5],
				"ref_ax": ref_state[6], "ref_ay": ref_state[7], "ref_az": ref_state[8]
			})
		if self.is_record_obs:
			if obs_state.ndim != 1 or obs_state.size != 3:
				raise ValueError(f"Observer state dimension error, expected (9, ), got {obs_state.shape}")

			self.obs_data_log.append({
				"timestamp": sim_t,
				"obs_dx": obs_state[0], "obs_dy": obs_state[1], "obs_dz": obs_state[2]
			})

	def shutdown_handler(self):
		"""
		Handle data saving when node shuts down (with zip compression support)
		"""
		try:
			if len(self.data_log) == 0:
				rospy.loginfo("No data to save")
				return

			# 修正控制名称类型检查（原逻辑错误）
			if not isinstance(self.control_name, str) or self.control_name.strip() == "":
				rospy.logerr("Controller name is invalid or empty")
				return

			save_chioce = input("\nData recording completed. Enter '1' to save data, '0' to discard: ")
			while save_chioce not in ['0', '1']:
				save_chioce = input("Invalid input. Please enter '1' to save or '0' to discard: ")
			
			if save_chioce == '1':
				data_dir = Filepath("scripts/data")
				data_dir.mkdir(exist_ok=True)
				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				base_name = f"{timestamp}_{self.control_name}"

				# 保存原始数据文件
				file_paths = []
				# UAV状态数据（必存）
				uav_path = data_dir / f"{base_name}_uav_data.csv"
				pd.DataFrame(self.data_log).to_csv(uav_path, index=False)
				file_paths.append(uav_path)
				
				# 参考数据（可选）
				if len(self.ref_data_log) != 0:
					ref_path = data_dir / f"{base_name}_ref_data.csv"
					pd.DataFrame(self.ref_data_log).to_csv(ref_path, index=False)
					file_paths.append(ref_path)
				
				# 观测数据（可选）
				if len(self.obs_data_log) != 0:
					obs_path = data_dir / f"{base_name}_obs_data.csv"
					pd.DataFrame(self.obs_data_log).to_csv(obs_path, index=False)
					file_paths.append(obs_path)

				rospy.loginfo(f"Data saved to: {[str(p) for p in file_paths]}")

				# 询问是否压缩
				zip_choice = input("Do you want to compress these files into a zip? (0/1): ").strip().upper()
				while zip_choice not in ['0', '1']:
					zip_choice = input("Invalid input. Please enter '1' to compress or '0' to skip: ").strip().upper()

				if zip_choice == '1':
					zip_path = data_dir / f"{base_name}_data.zip"
					try:
						print('save')
						import zipfile
						with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
							for file in file_paths:
								if file.exists():  # 确保文件存在再压缩
									zf.write(file, arcname=file.name)  # 保留文件名
						rospy.loginfo(f"Successfully compressed to: {zip_path}")
						
						# 可选：是否删除原始文件（根据需求决定，这里注释掉保留原始文件）
						for file in file_paths:
							file.unlink(missing_ok=True)
					except Exception as e:
						rospy.logerr(f"Failed to create zip file: {str(e)}")
			else:
				rospy.loginfo("Data discarded")

		except Exception as e:
			rospy.logerr(f"Error during shutdown handling: {str(e)}")
    
	def reach_target_point_by_pid(self):
		rospy.loginfo(f"Moving to target position: {self.target_position} by pid")


	def run(self, attitude_controller, simulation_time: float):
		"""
		Main control loop
		:param attitude_controller: Outer-loop controller function, returns (phi_d, theta_d, psi_d, thrust)
		:param simulation_time: Simulation duration (s)
		"""
		
		self.initialize_system()
		rospy.loginfo(f"Control loop started at {1/self.dt:.1f}Hz")
		rospy.loginfo("Starting position guidance...")

		self.reach_target_position()
		rospy.loginfo("Waiting for 1 second...")
		for _ in range(100):
			self.pub_position_keep()
			self.rate.sleep()

		rospy.loginfo("Switching to attitude control...")

		self.is_record_ref = True
		self.is_show_rviz = True

		start_time = rospy.Time.now()
		current_time = start_time
		self.t0 = start_time.to_sec()

		while (not rospy.is_shutdown()) and ((current_time - start_time).to_sec() < simulation_time):
			current_time = rospy.Time.now()            
			phi_d, theta_d, psi_d, thrust = attitude_controller()           # Get controller output
			self.publish_attitude_command(phi_d, theta_d, psi_d, thrust)    # Publish control commands
			self.data_record(sim_t=(current_time - start_time).to_sec(), ref_state=np.zeros(9))
			self.rate.sleep()
		self.is_show_rviz = False
		self.target_position = (0.0, 0.0, 0.5)
		self.reach_target_position()

def example_controller():
    """Example controller: Hover command (returns desired attitude and thrust)"""
    return (0.0, 0.0, 0.0, 0.75*9.8)  # (roll, pitch, yaw, thrust)

# if __name__ == "__main__":
#     controller = UAV_ROS()
#     controller.run(example_controller, simulation_time=10.0)
