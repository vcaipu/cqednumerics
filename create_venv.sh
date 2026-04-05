pip install --upgrade pip
python3 -m venv jax-gpu
source jax-gpu/bin/activate

pip install jupyter ipykernel
# One install keeps jax and jaxlib on the same release (avoid jax/jaxlib mismatch).
pip install --upgrade "jax[cuda12]"
pip install matplotlib scikit-fem jaxopt meshio
