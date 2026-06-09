
import sys
sys.path.insert(0, '..')
import main
import psutil
print("=== Testing System Metrics ===")
print("\n1. CPU Usage:", psutil.cpu_percent(interval=1))
temp = main.get_cpu_temperature()
print(f"2. CPU Temperature: {temp}°C" if temp is not None else "2. CPU Temperature: Not available")
gpu = main.get_gpu_stats()
print(f"3. GPU Stats: {gpu}")
