## Installation

### Install RGB-T detection

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n RGBTDet python=3.7 -y
source activate RGBTDet

conda install cython
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the repository.

```shell
git clone https://github.com/ccccwb/Multimodal-Detection-and-Tracking-UAV.git
cd Multimodal-Detection-and-Tracking-UAV/RGBT-Detection
```

d. Compile BboxToolkit extensions.

```shell
cd BboxToolkit
pip install -e .
cd ..
```

e. Compile MultiScaleDeformableAttention extensions.

```shell
cd MultiScaleDeformableAttention
pip install -e .
cd ..
```

f. Install RGBTDet (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```
