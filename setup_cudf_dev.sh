#!/bin/bash

set -e

echo "🚀 Bootstrapping cuDF dev environment..."

# -------------------------------
# 1. System deps
# -------------------------------
apt-get update
apt-get install -y gcc-12 g++-12

# -------------------------------
# 2. CUDA compiler fix
# -------------------------------
export CC=gcc-12
export CXX=g++-12
export CUDAHOSTCXX=g++-12
export CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12

# persist (optional but nice)
echo 'export CC=gcc-12' >> ~/.bashrc
echo 'export CXX=g++-12' >> ~/.bashrc
echo 'export CUDAHOSTCXX=g++-12' >> ~/.bashrc
echo 'export CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12' >> ~/.bashrc

# -------------------------------
# 3. Conda env
# -------------------------------
source /opt/conda/etc/profile.d/conda.sh

conda create -n cudf-dev -y \
  -c rapidsai -c conda-forge -c nvidia \
  cudf=26.2 python=3.10 cuda-version=12.2

conda activate cudf-dev

# -------------------------------
# 4. Install editable cudf
# -------------------------------
echo "🔧 Installing local cudf..."

pip uninstall cudf-cu12 -y || true
pip install -e python/cudf --no-deps

# -------------------------------
# 5. Verify
# -------------------------------
python - <<EOF
import cudf
print("cuDF version:", cudf.__version__)
print("cuDF path:", cudf.__file__)
EOF

echo "✅ Environment ready. Keep cooking."