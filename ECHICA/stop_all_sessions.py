import os
import signal
import subprocess
import sys

# Windows-specific process killing

def kill_processes_by_name(name):
    try:
        # Use tasklist to find all processes
        tasks = subprocess.check_output(['tasklist'], shell=True).decode()
        for line in tasks.splitlines():
            if name.lower() in line.lower():
                pid = int(line.split()[1])
                os.kill(pid, signal.SIGTERM)
                print(f"Killed {name} process with PID {pid}")
    except Exception as e:
        print(f"Error killing {name}: {e}")

if __name__ == "__main__":
    # Kill all python processes running echica_model.py
    kill_processes_by_name('echica_model.py')
    # Kill all streamlit processes
    kill_processes_by_name('streamlit.exe')
    kill_processes_by_name('streamlit')
    # Kill all ngrok processes
    kill_processes_by_name('ngrok.exe')
    kill_processes_by_name('ngrok')
    print("All relevant sessions stopped.") 