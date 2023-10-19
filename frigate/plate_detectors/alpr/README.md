# SmartCam_LPR

![apm](https://img.shields.io/apm/l/vim-mode.svg)

Automatically detect license plate (bounding boxes and its 4 corners) with RetinaPlate, then recognize its character using LPRNet

# Introduction

## Performance

|Dataset Name|Accuracy|Download|
|---|---|---|
|Indoor Test|99.76%|[Detect](http://192.168.0.222/cannn/smartcan_lpr/-/tree/master/retina_plate%2Fweight) / [OCR](http://192.168.0.222/cannn/smartcan_lpr/-/tree/master/LPRnet)|
| Test|79.61%| |

## Dependencies
- Python 3.6.8
  
Install environments for Tensorflow GPU is a painful task. Pls strictly follow below steps

Install anaconda, then create new environment by:
```bash
conda create -n alpr python==3.6
conda activate alpr
```
- Install tensorflow-gpu 1.14 with conda. It automatically installs cuda 10.0.130 and cudnn 7.6.5
```bash
conda install tensorflow-gpu=1.13
```
- Install pytorch, which competitive with cuda 10.0
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

```bash
pip install -r requirement.txt
```
## Usage

### Data preprocess
Get data:
```bash
git clone http://192.168.0.222/cannn/smartcan_lpr.git
```
### Run samplex
To predict an image
```bash
python main.py
```
To predict a video
```bash
python main_video.py
```
To use a live rtsp video
```bash
python run_threading.py
```
## Parameter for DLC/Java execution

|Model|Input name|Output name|Layer name| Input Size|
|---|---|---|---|---|
|RetinaPlate|input0|conf0 / landmark0 / loc0|Concat_223 / Concat_248 / Concat_198| 480x850x3
|OCR LPRnet|inputs|mean_out|-|24x94x3|
