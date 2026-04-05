import subprocess
import numpy as np

separations = np.arange(1,31,3)
sep = 30
Ns = np.arange(1000,10000,1000)

for N in Ns:
    try:
        print(f"##### RUNNING N = {N} #############")
        command = ['python3',"./2D/2DZeroPoint.py",f"--plotdir=./2D/phasetransitiontry4/{N}/",f"--separation={sep}",f"--N={N}",f"--n=50"]
        result = subprocess.run(command, check=True)
        
        # The return code is stored in the CompletedProcess object
        print(f"Command exited with return code: {result.returncode}")

    except FileNotFoundError as e:
        print(f"Error: Command not found. Details: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}. Output:\n{e.output}")