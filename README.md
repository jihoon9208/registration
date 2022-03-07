# registration

pytorch 10.2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


pip install numpy scipy matplotlib open3d tensorboardX future-fstrings easydict joblib learn

pip install future python-igraph tqdm pytorch3d

export CUDA_HOME=/usr/local/cuda-11.3

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"


conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv

conda install faiss-gpu cudatoolkit=10.0 -c pytorch
sudo apt-get install libopenblas-dev
sudo apt-get install libomp-dev

CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.7m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3   
make