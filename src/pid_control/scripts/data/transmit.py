import os
import subprocess


def find_latest_zip(data_dir):
    """查找目录下最新的 _data.zip 文件（按修改时间排序）"""
    zip_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('_data.zip') and os.path.isfile(os.path.join(data_dir, f))
    ]
    if not zip_files:
        raise FileNotFoundError(f"No _data.zip files found in {data_dir}")
    
    # 按修改时间降序排序，取第一个（最新）
    zip_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return zip_files[0]


def scp_transfer(local_file, remote_host, remote_port, remote_user, remote_dir):
    """
    通过 scp 命令传输文件到远程主机
    :param local_file: 本地文件路径（绝对路径）
    :param remote_host: 远程主机 IP/域名
    :param remote_port: SSH 端口（默认 22）
    :param remote_user: 远程用户名
    :param remote_dir: 远程目标目录（需提前存在）
    """
    # 构造 scp 命令（格式：scp -P 端口 本地文件 用户名@主机:远程目录）
    remote_path = f"{remote_user}@{remote_host}:{remote_dir}"
    cmd = f"scp -P {remote_port} {local_file} {remote_path}"

    try:
        # 执行 scp 命令（check=True 表示命令失败时抛异常）
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Transfer successful!\nLocal file: {local_file}\nRemote path: {remote_dir}")
        return True

    except subprocess.CalledProcessError as e:
        # 命令执行失败（如认证错误、网络问题、目录不存在等）
        error_msg = e.stderr.strip()
        if "No such file or directory" in error_msg:
            print(f"Error: Remote directory {remote_dir} does not exist!")
        elif "Permission denied" in error_msg:
            print(f"Error: Authentication failed (check password or permissions)")
        else:
            print(f"scp command failed: {error_msg}")
        return False

    except Exception as e:
        print(f"Unexpected error during transfer: {str(e)}")
        return False


if __name__ == "__main__":
    # ==================== 用户配置（根据实际情况修改） ====================
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # 本地数据目录（同之前代码）
    REMOTE_HOST = "192.168.50.203"                        # 远程主机 IP
    REMOTE_PORT = 22                                      # SSH 端口（默认 22）
    REMOTE_USER = "ht"                                  # 远程用户名
    REMOTE_DIR = "/home/ht/src/px4_ws/src/pid_control/scripts/data"  # 远程目标目录
    # =====================================================================

    # 查找最新的 ZIP 文件
    try:
        latest_zip = find_latest_zip(DATA_DIR)
        print(f"Found latest ZIP file: {latest_zip}")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 执行传输
    transfer_success = scp_transfer(
        local_file=latest_zip,
        remote_host=REMOTE_HOST,
        remote_port=REMOTE_PORT,
        remote_user=REMOTE_USER,
        remote_dir=REMOTE_DIR
    )

    if not transfer_success:
        exit(1)