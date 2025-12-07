pip install --upgrade pip
python3 -m venv jax-gpu
source jax-gpu/bin/activate

pip install jupyter ipykernel
pip install jax
pip install -U "jax[cuda12]"
pip install matplotlib scikit-fem jaxopt meshio
