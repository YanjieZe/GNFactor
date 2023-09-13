
# Error Catching

- PyRender error.

add following to bashrc:
```
export DISPLAY=:0
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
```


- `qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "xxxx/cv2/qt/plugins`
```
pip uninstall opencv-python (this is very important!)
pip install opencv-python-headless
```

- xcb errors during generating data
```
sudo apt-get install qt5-default
```

- import transformers error: `packaging.version.InvalidVersion: Invalid version: '0.10.1,<0.11'`
```
pip install packaging==21.3
```

- `AssertionError: Invalid type <class 'NoneType'> for key WEIGHT_DECAY_BIAS; valid types = {<class 'tuple'>, <class 'int'>, <class 'float'>, <class 'bool'>, <class 'list'>, <class 'str'>}`
```
pip uninstall yacs
pip install yacs --upgrade
```

- `ImportError: cannot import name 'SCMode' from 'omegaconf' (/home/yanjieze/miniconda3/envs/vision/lib/python3.8/site-packages/omegaconf/__init__.py)`
```
pip uninstall omegaconf
pip install omegaconf==2.1.1
```

- `ImportError: cannot import name 'VisionEncoderDecoderModel' from 'transformers' (unknown location)`

```
pip uninstall transformers
pip install transformers==4.20.1
```

- `AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)`
```
pip install opencv-python==4.5.5.64
```

- `ImportError: cannot import name 'get_num_classes' from 'torchmetrics.utilities.data' `
```
pip install torchmetrics==0.5.1
```

- `assert mdl is not None`
```
pip install hydra-core==1.1
```

- `MultiScaleDeformableAttention....` error
```
cd ~/miniconda3/envs/nerfact/lib/python3.9/site-packages
rm -rf MultiScaleDeformableAttention.cpython-39-x86_64-linux-gnu.so 
```

- `qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "/data/yanjieze/miniconda3/envs/nerfact/lib/python3.9/site-packages/cv2/qt/plugins`
```
pip uninstall opencv-python
pip install opencv-python-headless
```

