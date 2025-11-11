# This code autostarts grpo-mlx.py to clear out allocated space due to memory leaking
import subprocess
import time

# Duration to run the command (in seconds)
RUN_DURATION = 30 * 60

while True:
    try:
        start_time = time.time()
        
        # Run the command and capture output in real-time
        process = subprocess.Popen(["python3", "-m" "grpo.grpo-trl"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor the process for the specified duration
        while time.time() - start_time < RUN_DURATION:
            output = process.stdout.readline()
            if output:
                print(output, end="", flush=True)  # Print output in real-time
            
            if process.poll() is not None:  # Check if process has exited early
                print("Process terminated unexpectedly. Restarting...", flush=True)
                break
            time.sleep(0.1)  # Check every 0.1 second
        
        # Stop the process after 45 minutes if it's still running
        if process.poll() is None:
            print("Stopping process after 45 minutes.")
            process.terminate()
            process.wait()
        
    except Exception as e:
        print(f"Error encountered: {e}. Restarting process...", flush=True)
    
    print("Restarting training process...", flush=True)
    time.sleep(5)  # Short pause before restarting
