# registration
python3.9

pytorch 10.2
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
y
conda install pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch -c nvidia

pip install torch==1.10.0+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html


pip install numpy scipy matplotlib open3d tensorboardX future-fstrings easydict joblib learn

pip install python-igraph tqdm pytorch3d
conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; 
pip install torch ninja

export CUDA_HOME=/usr/local/cuda-11.0
conda install openblas-devel -c anaconda

pip install MinkowskiEngine==0.5.3 -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"




conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; 

conda install -c r libiconv

conda install faiss-gpu cudatoolkit=11.0 -c pytorch-gpu


sudo apt-get install libopenblas-dev
sudo apt-get install libomp-dev

CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION



cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.7m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3   

cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.8.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.8 -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3   
make

----------------------------------------------------------------------------

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install numpy
conda install openblas-devel -c anaconda

pip install MinkowskiEngine==0.5.3 -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install rich scikit-image matplotlib imageio plotly opencv-python

conda install -c conda-forge faiss-gpu cudatoolkit=11.3 

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install jupyter
conda install pytorch3d -c pytorch3d -> torch version 1.10.1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install scipy matplotlib open3d tensorboardX future-fstrings easydict joblib learn 

pip install python-igraph 

https://github.com/KinglittleQ/torch-batch-svd.git

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh


KITTI learning


python main.py --dataset 