# Facial-Recognition
:camera: Simple Python implementation using OpenCV


## Installation

Clone the git repo :
```sh
git clone https://github.com/axelbellec/facial-recognition.git
```

```sh
pip install -r requirements.txt
```

#### Virtual environment for OpenCV with [Anaconda](https://www.continuum.io/downloads)

```sh
conda create -n opencv numpy scipy scikit-learn matplotlib python=3
```

`conda create` create a new virtual environment. `-n` let us specify the environment name (`opencv`). We add packages we want to install. Then, we add `python=3` because we want to use Python 3.

Activate it with:
```sh
source activate opencv
```

#### Install OpenCV 3

Once we launch the virtual environment, we can download OpenCV 3 through *binstar*:
```
conda install -c https://conda.binstar.org/menpo opencv3
```

OpenCV3 is now installed. We just need to check if Python 
achieve to communicate with this library.

#### Test installation

```python
import cv2
print(cv2.__version__)
```

You should see:
```
3.1.0
```

## Usage

```
python cam_detection.py --help

Usage: cam_detection.py [OPTIONS]

Options:
  --eyes BOOLEAN       add eyes detection
  --face_cascade PATH  frontal face classifier
  --eyes_cascade PATH  eyes classifier
  --help               Show this message and exit.
```

You can add your own classifiers.

```
python cam_detection.py --face_cascade <face_cascade_classifier> --eyes True --eyes_cascade <eyes_cascade_classifier>
```