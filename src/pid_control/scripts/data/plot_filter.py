import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import os

# Get the absolute path of the current script (critical: define data directory as script's location)
data_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
os.makedirs(data_dir, exist_ok=True)  # Create directory (optional, ensure directory exists)

print(f"Data directory: {data_dir}")  # Print data directory for debugging

filter_files = [
    os.path.join(data_dir, f)  # Concatenate full path
    for f in os.listdir(data_dir)
    if f.endswith('_uav_data.csv')
]

print("Filter files:", [os.path.basename(f) for f in filter_files]) 

filter_files.sort()

filter_filename = filter_files[-1]
filter_data = pd.read_csv(filter_filename)

time = filter_data['Timestamp'].to_numpy()
filter_x = filter_data['filtered_x'].to_numpy()
filter_y = filter_data['filtered_y'].to_numpy()
filter_z = filter_data['filtered_z'].to_numpy()
filter_vx = filter_data['filtered_vx'].to_numpy()
filter_vy = filter_data['filtered_vy'].to_numpy()
filter_vz = filter_data['filtered_vz'].to_numpy()

measure_x = filter_data['measure_x'].to_numpy()
measure_y = filter_data['measure_y'].to_numpy()
measure_z = filter_data['measure_z'].to_numpy()
measure_vx = filter_data['measure_vx'].to_numpy()
measure_vy = filter_data['measure_vy'].to_numpy()
measure_vz = filter_data['measure_vz'].to_numpy()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(321)
ax1.plot(time, filter_x, label="filter")
ax1.plot(time, measure_x, label="measure")
ax1.set_xlabel('X(m)')
ax1.set_ylabel('time')
ax1.legend()
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax2 = fig.add_subplot(322)
ax2.plot(time, filter_y, label="filter")
ax2.plot(time, measure_y, label="measure")
ax2.set_xlabel('Y(m)')
ax2.set_ylabel('time')
ax2.legend()
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax3 = fig.add_subplot(323)
ax3.plot(time, filter_z, label="filter")
ax3.plot(time, measure_z, label="measure")
ax3.set_xlabel('Z(m)')
ax3.set_ylabel('time')
ax3.legend()
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax4 = fig.add_subplot(324)
ax4.plot(time, filter_vx, label="filter")
ax4.plot(time, measure_vx, label="measure")
ax4.set_xlabel('VX')
ax4.set_ylabel('time')
ax4.legend()
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax5 = fig.add_subplot(325)
ax5.plot(time, filter_vy, label="filter")
ax5.plot(time, measure_vy, label="measure")
ax5.set_xlabel('VY(m)')
ax5.set_ylabel('time')
ax5.legend()

ax6 = fig.add_subplot(326)
ax6.plot(time, filter_vz, label="filter")
ax6.plot(time, measure_vz, label="measure")
ax6.set_xlabel('VZ(m)')
ax6.set_ylabel('time')
ax6.legend()

plt.tight_layout()
plt.show()

