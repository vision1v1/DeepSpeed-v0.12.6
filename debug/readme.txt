# 需要安装 mpi4py 参考 https://mpi4py.readthedocs.io/en/latest/install.html#linux
conda install -c conda-forge mpi4py openmpi



# 需要安装 ninja , 参考 https://www.claudiokuenzler.com/blog/756/install-newer-ninja-build-tools-ubuntu-14.04-trusty#.XEDUk89KjOB

# 第一步
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
# 测试
/usr/bin/ninja --version

# 第二步
如果已经安装了ninja , 先卸载ninja 
pip uninstall ninja
pip install ninja


# 安装 deepspeed 参考 https://www.deepspeed.ai/tutorials/advanced-install/
