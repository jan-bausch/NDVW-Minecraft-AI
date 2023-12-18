import subprocess
import multiprocessing
import sys
import time

def run_exe(instance_id, exe_path):
    while True:
        try:
            print(f"Starting instance {instance_id}")
            subprocess.run([exe_path])
            print(f"Instance {instance_id} crashed. Restarting...")
        except Exception as e:
            print(f"Exception occurred in instance {instance_id}: {e}")

if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    exe_path = sys.argv[2]
    
    processes = []
    for i in range(num_instances):
        process = multiprocessing.Process(target=run_exe, args=(i, exe_path))
        process.start()
        processes.append(process)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
