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
module load cudatoolkit/13.0
conda activate [name of conda package]
```

The packages you need are: 
```
pip install jax
pip install -U "jax[cuda13]"
```

## Step 3: Activate GPU Interactive Session and Connect Jupyter Kernel

Activate session by: 
```
salloc --time=30:00  --nodes=1  --ntasks-per-node=1  --cpus-per-task=4 --mem=20G  --gres=gpu:1 --qos=gpu-short
```

Or from Della online docs: 
```
salloc --nodes=1 --ntasks=1 --mem=4G --time=00:20:00 --gres=gpu:1 --mail-type=begin
```

Then follow steps below to go to GPU Node. Will NOT be in the login node. 

## Connecting to a GPU Interactive Session: 

Put the following in a ssh config file like `~/.ssh/config` on your local machine:

```
Host della-login
    HostName della.princeton.edu
    User <your_netid>

Host della-l* # Matches any compute node starting with della-l
    ProxyJump della-login
    User <your_netid>
```

So that it can redirect properly to della GPU node. 

Then, look at the node name in the salloc-ed GPU node (something like della-lXXX), and use command palette to SSH into that by just typing as hostname "netid@della-lXXX"

--------------

Then activate environment by `source start.sh`

Check can find GPU by running the test.py file, and seeing "gpu" on last line

Connect to Jupyter Kernel by installing Jupyter Kernel in Python virtual env. Then should just be able to select Kernel in VSCode (when you first run a cell, it will prompt you to select a python virtual environment.)