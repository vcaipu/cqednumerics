import pickle
import os
import argparse
import sys

# Try to find the venv and add its site-packages to sys.path
# Looking at the workspace, there is a create_venv.sh, so maybe there is a venv folder.
# Or we can try to guess where the libraries are.
# Common locations: venv/lib/python3.9/site-packages, etc.

def find_site_packages():
    # Check for common venv names
    for venv_name in ['venv', 'jax-gpu', '.venv']:
        base = os.path.join('/scratch/gpfs/AROD/vc9839/finite-island-cqed', venv_name)
        if os.path.exists(base):
            # Find the lib/pythonX.X/site-packages
            lib_dir = os.path.join(base, 'lib')
            if os.path.exists(lib_dir):
                for py_dir in os.listdir(lib_dir):
                    sp = os.path.join(lib_dir, py_dir, 'site-packages')
                    if os.path.exists(sp):
                        return sp
    return None

sp = find_site_packages()
if sp:
    sys.path.insert(0, sp)

# Also add workspace root
sys.path.insert(0, '/scratch/gpfs/AROD/vc9839/finite-island-cqed')

def format_time(seconds):
    if seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"

def get_run_data(run_dir):
    results_path = os.path.join(run_dir, "results.pkl")
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        
        params = data.get("parameters", {})
        ej = data.get("E_J", 0.0)
        ec = data.get("E_C", 1.0)
        
        # Calculate Volume Error
        vol = params.get("sidelenX", 0) * params.get("sidelenY", 0) * params.get("sidelenZ", 0) * 2
        int_vol = data.get("integrated_volume", 0)
        vol_err = ((int_vol - vol) / vol * 100) if vol != 0 else 0
        
        return {
            "Run": os.path.basename(run_dir),
            "EJ/EC": f"{ej/ec:.4f}" if ec != 0 else "inf",
            "EJ": f"{ej:.6f}",
            "EC": f"{ec:.6f}",
            "Time": format_time(data.get("totalTime")),
            "Sep": params.get("separation"),
            "Island (X,Y,Z)": f"({params.get('sidelenX')}, {params.get('sidelenY')}, {params.get('sidelenZ')})",
            "lc (L,S)": f"({params.get('lc_large')}, {params.get('lc_small')})",
            "Grid (X,Y,Z)": f"({params.get('gridlenX')}, {params.get('gridlenY')}, {params.get('gridlenZ')})",
            "DoF": data.get("femsystem").dofs if data.get("femsystem") else "N/A",
            "Vol Err": f"{vol_err:.2f}%"
        }
    except Exception as e:
        # print(f"Error reading {run_dir}: {e}")
        return None

def print_table(data):
    if not data:
        return
    headers = list(data[0].keys())
    widths = {h: len(h) for h in headers}
    for row in data:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))
    
    header_row = " | ".join(str(h).ljust(widths[h]) for h in headers)
    print(header_row)
    print("-" * len(header_row))
    for row in data:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))

def main():
    parser = argparse.ArgumentParser(description="Summarize run results from allplots/run*")
    parser.add_argument("--base", type=str, default="3D/allplots", help="Base directory containing run folders")
    parser.add_argument("--runs", type=int, nargs='+', help="Specific run numbers to include")
    args = parser.parse_args()

    table_data = []
    if args.runs:
        run_dirs = [os.path.join(args.base, f"run{r}") for r in args.runs]
    else:
        if not os.path.exists(args.base):
            print(f"Base directory {args.base} does not exist.")
            return
        run_dirs = [os.path.join(args.base, d) for d in os.listdir(args.base) if d.startswith("run")]
        run_dirs.sort(key=lambda x: int(os.path.basename(x).replace("run", "")) if os.path.basename(x).replace("run", "").isdigit() else 0)

    for run_dir in run_dirs:
        row = get_run_data(run_dir)
        if row:
            table_data.append(row)

    if not table_data:
        print("No results found. (Make sure you are running with the correct python interpreter that has skfem/jax installed)")
        return

    print_table(table_data)

if __name__ == "__main__":
    main()
