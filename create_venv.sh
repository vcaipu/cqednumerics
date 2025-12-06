python3 -m venv jax-gpu
source jax-gpu/bin/activate
pip install -r requirements.txt

pip install jupyter ipykernel
pip install jax
pip install -U "jax[cuda13]"
pip install matplotlib scikit-fem jaxopt meshio
