# PyTorch Transformer for RUL Prediction
An implementation with Transformer encoder and convolution layers with TensorFlow 2 / Keras for remaining useful life prediction (originally written in PyTorch).   
_Author: Jiaxiang Cheng, Nanyang Technological University, Singapore_

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" />

## Quick Run
Install TensorFlow 2 and the other dependencies listed under [Environment Details](#environment-details), then run `python train.py --dataset FD001`. And you will get the training loss and testing result for each epoch where the RMSE is from the test set:
```
Epoch: 0, loss: 9474.43470, RMSE: 61.11946
Epoch: 1, loss: 5858.27227, RMSE: 46.03318
Epoch: 2, loss: 3208.53410, RMSE: 29.78244
Epoch: 3, loss: 1310.71390, RMSE: 22.94705
...
```
The testing is conducted for each epoch as the data set is not large so it's no big deal but you may remove them and only do the evaluation after finishing the training epochs.

## Environment Details
```
python==3.8.8
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.1
tensorflow==2.13.0
```

## Credit
This work is inpired by Mo, Y., Wu, Q., Li, X., & Huang, B. (2021). Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit. Journal of Intelligent Manufacturing, 1-10.

## Citation
[![DOI](https://zenodo.org/badge/365902211.svg)](https://zenodo.org/badge/latestdoi/365902211)

## License
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
