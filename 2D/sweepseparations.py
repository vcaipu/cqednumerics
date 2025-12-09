import subprocess
import numpy as np

separations = np.arange(1,31,3)

for sep in separations:
    try:
        print(f"##### RUNNING SEPARATION = {sep} #############")
        command = ['python3',"./2D/2DZeroPoint.py",f"--plotdir=./2D/plots/{sep}/",f"--separation={sep}"]
        result = subprocess.run(command, check=True)
        
        # The return code is stored in the CompletedProcess object
        print(f"Command exited with return code: {result.returncode}")

    except FileNotFoundError as e:
        print(f"Error: Command not found. Details: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}. Output:\n{e.output}")