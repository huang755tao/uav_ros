import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit


# Zip file extraction code
def unzip_latest_data(data_dir):
    """Unzip the latest data.zip file in the directory"""
    # Find all zip files matching the naming convention (assumes zip files are in the format *_data.zip)
    zip_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('_data.zip') and os.path.isfile(os.path.join(data_dir, f))
    ]
    
    if not zip_files:
        print("Warning: No _data.zip files found, attempting to read CSV files directly")
        return False

    # Sort by modification time and take the latest zip file
    zip_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_zip = zip_files[0]
    print(f"Found latest zip file: {os.path.basename(latest_zip)}")

    try:
        # Unzip to the current data directory
        with zipfile.ZipFile(latest_zip, 'r') as zf:
            # Verify the zip contains the required three data files
            required_files = [f for f in zf.namelist() 
                            if f.endswith(('_uav_data.csv', '_ref_data.csv', '_obs_data.csv'))]
            if len(required_files) < 3:
                raise ValueError("Zip file is missing necessary data files (requires _uav/_ref/_obs ending CSVs)")
            
            # Execute extraction
            zf.extractall(path=data_dir)
            print(f"Successfully unzipped the following files to {data_dir}: {required_files}")
            return True

    except zipfile.BadZipFile:
        print(f"Error: Zip file is corrupted and cannot be unzipped: {latest_zip}")
        return False
    except PermissionError:
        print(f"Error: No permission to unzip file: {latest_zip} (check directory permissions)")
        return False
    except Exception as e:
        print(f"Unzip failed: {str(e)}")
        return False



# Get the absolute path of the current script (critical: define data directory as script's location)
data_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
os.makedirs(data_dir, exist_ok=True)  # Create directory (optional, ensure directory exists)

print(f"Data directory: {data_dir}")  # Print data directory for debugging

# New: Unzip the latest data.zip file (unzip first, then read CSV)
unzip_success = unzip_latest_data(data_dir)
if unzip_success:
    print("Zip unzipping completed, about to read the latest unzipped CSV files")
else:
    print("Failed to unzip zip file, attempting to read existing CSV files...")


# Find all valid CSV files (including full paths)
drone_files = [
    os.path.join(data_dir, f)  # Concatenate full path
    for f in os.listdir(data_dir)
    if f.endswith('_uav_data.csv')
]
reference_files = [
    os.path.join(data_dir, f)  # Concatenate full path
    for f in os.listdir(data_dir)
    if f.endswith('_ref_data.csv')
]
observer_files = [
    os.path.join(data_dir, f)  # Concatenate full path
    for f in os.listdir(data_dir)
    if f.endswith('_obs_data.csv')
]

print("Drone files:", [os.path.basename(f) for f in drone_files])  # Print file names (for debugging)
print("Reference files:", [os.path.basename(f) for f in reference_files])  # Print file names (for debugging)
print("Observer files:", [os.path.basename(f) for f in observer_files]) 

drone_files.sort()
reference_files.sort()
observer_files.sort()

if not drone_files or not reference_files:
    print("Error: No drone data or reference trajectory files found.")
else:
    drone_filename = drone_files[-1]  # Latest drone data file (with full path)
    reference_filename = reference_files[-1]  # Latest reference trajectory file (with full path)
    observer_filename = observer_files[-1] 

    try:
        # Read data (use full path)
        drone_data = pd.read_csv(drone_filename)
        reference_data = pd.read_csv(reference_filename)
        observer_data = pd.read_csv(observer_filename)
    except FileNotFoundError:
        print(f"Error: File not found\nDrone data file: {drone_filename}\nReference trajectory file: {reference_filename}")
        exit(1)

    # Extract data and convert to numpy arrays
    drone_time = drone_data['timestamp'].to_numpy()
    drone_x = drone_data['x'].to_numpy()
    drone_y = drone_data['y'].to_numpy()
    drone_z = drone_data['z'].to_numpy()
    drone_vx = drone_data['vx'].to_numpy()
    drone_vy = drone_data['vy'].to_numpy()
    drone_vz = drone_data['vz'].to_numpy()

    reference_time = reference_data['timestamp'].to_numpy()
    reference_x = reference_data['ref_x'].to_numpy()
    reference_y = reference_data['ref_y'].to_numpy()
    reference_z = reference_data['ref_z'].to_numpy()
    reference_vx = reference_data['ref_ax'].to_numpy()
    reference_vy = reference_data['ref_ay'].to_numpy()
    reference_vz = reference_data['ref_az'].to_numpy()

    obs_time = observer_data['timestamp'].to_numpy()
    obs_x = observer_data['obs_dx'].to_numpy()
    obs_y = observer_data['obs_dy'].to_numpy()
    obs_z = observer_data['obs_dz'].to_numpy()

    # Calculate errors
    error_x = np.abs(drone_x - reference_x)
    error_y = np.abs(drone_y - reference_y)
    error_z = np.abs(drone_z - reference_z)

    error_vx = np.abs(drone_vx - reference_vx)
    error_vy = np.abs(drone_vy - reference_vy)
    error_vz = np.abs(drone_vz - reference_vz)

    # Plot trajectories
    fig = plt.figure(figsize=(12, 12))

    # 2D trajectory plot
    ax1 = fig.add_subplot(331)
    ax1.plot(reference_x, reference_y, label='Reference Trajectory', color='blue', linestyle='--')  # Add reference trajectory
    ax1.plot(drone_x, drone_y, label='Drone Trajectory', color='red')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()

    # 3D trajectory plot
    ax2 = fig.add_subplot(332, projection='3d')
    ax2.plot(reference_x, reference_y, reference_z, label='Reference Trajectory', color='blue', linestyle='--')  # Add reference trajectory
    ax2.plot(drone_x, drone_y, drone_z, label='Drone Trajectory', color='red')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Z Position (m)')
    ax2.set_title('3D Trajectory Comparison')
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 1.5)
    ax2.legend()

    # Error curves (with time alignment, assuming timestamps are consistent, otherwise interpolation is needed)
    ax3 = fig.add_subplot(333)
    ax3.plot(drone_time, error_x, label='X Error', color='red')
    ax3.plot(drone_time, error_y, label='Y Error', color='green')
    ax3.plot(drone_time, error_z, label='Z Error', color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Error Curves')
    ax3.legend()

    # Position curves (with time alignment, assuming timestamps are consistent, otherwise interpolation is needed)
    ax4 = fig.add_subplot(334)
    ax4.plot(drone_time, drone_x, label='X', color='red')
    ax4.plot(drone_time, reference_x, label='X REF', color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('X Position Curves')
    ax4.legend()

    ax5 = fig.add_subplot(335)
    ax5.plot(drone_time, drone_y, label='Y', color='red')
    ax5.plot(drone_time, reference_y, label='Y REF', color='green')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('Y Position Curves')
    ax5.legend() 

    ax6 = fig.add_subplot(336)
    ax6.plot(drone_time, drone_z, label='Z', color='red')
    ax6.plot(drone_time, reference_z, label='Z REF', color='green')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title('Z Position Curves')
    ax6.legend()

    # Velocity curves (with time alignment, assuming timestamps are consistent, otherwise interpolation is needed)
    ax7 = fig.add_subplot(337)
    ax7.plot(drone_time, reference_vx, label='X REF Velocity', color='green')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Velocity (m/s)')  # Corrected unit to m/s
    ax7.set_title('X Velocity Curves')
    ax7.legend()

    ax8 = fig.add_subplot(338)
    ax8.plot(drone_time, reference_vy, label='Y REF Velocity', color='green')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Velocity (m/s)')  # Corrected unit to m/s
    ax8.set_title('Y Velocity Curves')
    ax8.legend() 

    ax9 = fig.add_subplot(339)
    ax9.plot(drone_time, reference_vz, label='Z REF Velocity', color='green')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Velocity (m/s)')  # Corrected unit to m/s
    ax9.set_title('Z Velocity Curves')
    ax9.legend()

    # Observer data plot
    fig2 = plt.figure(2)
    ax11 = fig2.add_subplot(131)
    ax11.plot(obs_time, obs_x)
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Observed X (m)')
    ax11.set_title('Observer X Data')

    ax12 = fig2.add_subplot(132)
    ax12.plot(obs_time, obs_y)
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Observed Y (m)')
    ax12.set_title('Observer Y Data')

    ax13 = fig2.add_subplot(133)
    ax13.plot(obs_time, obs_z)
    ax13.set_xlabel('Time (s)')
    ax13.set_ylabel('Observed Z (m)')
    ax13.set_title('Observer Z Data')

    plt.tight_layout()
    plt.show()
