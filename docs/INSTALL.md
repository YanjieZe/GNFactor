# Installing Conda Environment for GNFactor

The following guidance works well for a machine with 3090 GPU and cuda 11.4, a machine with A100 GPU and cuda 11.7, and more machines.

If you encounter any problems, please feel free to open an issue.

# 0 create python/pytorch env
```
conda remove -n gnfactor --all
conda create -n gnfactor python=3.9
conda activate gnfactor
```

## if cuda version <=11.3
```
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## if cuda version >=11.4 (11.7 here works for 11.4 actually)
```
conda install pytorch==1.10.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

# 1 install pytorch3d
```
cd ..
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -e .
cd ../GNFactor
```

# 2 install CLIP
```
cd ..
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ..
cd GNFactor

pip install open-clip-torch
```

# 3 download coppeliasim 
```
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz --no-check-certificate

tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz

rm CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz
```

# 4 add following lines to your `~/.bashrc` file. 
Remember to source your bashrc (source ~/.bashrc) and reopen a new terminal then.

You should replace the path here with your own path to the coppeliasim installation directory.
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

# 5 install PyRep
You should open a new terminal here, to make your .bashrc work.
```
cd third_party/PyRep
pip install -r requirements.txt
pip install .
cd ../..
```

# 6 install RLBench
```
cd third_party/RLBench
pip install -r requirements.txt
python setup.py develop
cd ../..
```

# 7 install YARR
```
cd third_party/YARR
pip install -r requirements.txt
python setup.py develop
cd ../..
```

# 8 install GNFactor
```
cd GNFactor
pip install -r requirements.txt
python setup.py develop
cd ..
```

# 9 install other utility packages
```
pip install packaging==21.3 dotmap pyhocon wandb chardet opencv-python-headless gpustat ipdb visdom sentencepiece
```

# 10 install odise
Install xformers (this version is a must to avoid errors from detectron2)
```
pip install xformers==0.0.18
```
Install Stable Diffusion
```
pip install stable-diffusion-sdkit==2.1.3
```
Install detectron2:
```
cd ..
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ../GNFactor
```
Install ODISE packages
```
cd third_party/ODISE
pip install -e .
cd ..
```

# 11 fix some possible problems
Since a lot of packages are installed, there are some possible bugs. Use these commands first before running the code.
```
pip install torchvision --upgrade
pip install hydra-core==1.1
pip install opencv-python-headless
pip install numpy==1.23.5
```


# Congratulations! You have successfully installed GNFactor!
Now, you should be able to run our training and evaluation scripts.

Please make sure you could both train and evaluate algorithms before conducting more experiments.

For possible errors, see [ERROR_CATCH.md](ERROR_CATCH.md). Don't hesitate to open an issue if you encounter any hard problems.


