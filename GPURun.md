# How to Run with GPU

## Step 1: Log onto Della

Using VSCode's "Remote - SSH" (from Microsoft) Extension: 
1. Install the extension
2. Open Command Palette (Cmd + Shit + P) and select option "Remote SSH: Connect to Host"
3. Select "della-gpu.princeton.edu" or "della-pli.princeton.edu" or whichever one to use.

Using terminal: 
1. Simply "ssh netid@della-gpu.princeton.edu"

Then go into /scratch/gpfs/netid

## Step 2: Load in Packages

Run in terminal, or in a script: 

```
module purge
module load cudatoolkit/12.6   anaconda3/2024.6
conda activate [name of conda package]
```
