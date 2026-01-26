import subprocess
import numpy as np

separations = np.arange(1,21,3)

### IMPORTANT: DIRECTORY TO SAVE RESULTS. MUST END WITH A SLASH /
save_dir = "./allplots/sweep1/"
material = 0.0106

for sep in separations:
    try:
        print(f"##### RUNNING Separation = {sep} #############")
        command = ['python3',"./3DZeroPointCG.py",f"--plotdir={save_dir}Separation {sep}/",f"--separation={sep}",f"--material={material}"]
        result = subprocess.run(command, check=True)
        
        # The return code is stored in the CompletedProcess object
        print(f"Command exited with return code: {result.returncode}")

    except FileNotFoundError as e:
        print(f"Error: Command not found. Details: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}. Output:\n{e.output}")