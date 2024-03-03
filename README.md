# Multimodal-Detection-and-Tracking-UAV
A Multimodal Detection and Tracking System based on DJI Payload SDK and Mobile SDK. 
![video](test_video.gif)

**🎉Our paper on "Modality Balancing Mechanism for RGB-Infrared Object Detection in Aerial Image" has been accepted at PRCV 2023! Find the code release on  [RGBT-Detection](./RGBT-Detection/README.md).**


## 👀Supported Features:

- &#9745; RGB + Thermal image support
- &#9745; Vehicle detection and tracking
- &#9745; Hardware-accelerated video decoding (nvmpi)
- &#9745; TensorRT integration

## 📖How to use
### 📦Prepare the detection model
Follow the instructions in the [RGBT-Detection documentation](./RGBT-Detection/README.md) to train the multimodal detection model.
### 💾Environment
Our drone is a DJI M300-RTK equipped with the Zenmuse H20T camera. The onboard AI computer is the Nvidia Jetson NX.

Our environment: Ubuntu 18.04, CUDA 11.2 
### 📚Install necessary libraries for PSDK
#### 1. Opencv 4.5.4
```
cd opencv-4.5.4
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.4/modules \
      -D BUILD_EXAMPLES=ON ..
# If memory error occurs, retry without using j6
sudo make -j6 
sudo make -j6 install
```
#### 2. NVMPI for hardware-accelerated video decoding
```
cd jetson-ffmpeg
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig

git clone git://source.ffmpeg.org/ffmpeg.git -b release/4.2 --depth=1
cd ffmpeg
wget https://github.com/jocover/jetson-ffmpeg/raw/master/ffmpeg_nvmpi.patch
git apply ffmpeg_nvmpi.patch
./configure --enable-nvmpi
make
```
### 📈Compile PSDK and run
```
cd PSDK-SYSU
mkdir build
cd build
cmake ..
make
# Run the executable
./bin/610_NX
```


## 👏Acknowledgment
This repo is based on DJI SDK, MMDetection... We are very grateful for their excellent work.
