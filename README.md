# ZLFSR

################################################################

This Project is a Pytorch implementation of "Learning Zero-shot dense light field reconstruction from heterogeneous imaging"

## Requirements
** Python 3.7

** PyTorch 1.2.0

** CUDA 10.0

** Cudnn

** Matlab 2017b (for training/test data generation and evaluation)


## Usage
** For simulation data
1. **Dataset** Please download the dataset from http://lightfields.stanford.edu/mvlf/ (i.e., 3vlf_lfr.tar)

2. **Data generation** Run `ZLFSR_simulation/matlab/PrepareData.m` to generate training data.

3. **Training** Run `ZLFSR_simulation/train.py` to perform network training.

4. **Test** Run `ZLFSR_simulation/test.py` to perform network testing.


** For real data
1. **Dataset** Please download the dataset via Baidu Drive：https://pan.baidu.com/s/1Mekzz8M8Y7A1QSDwpxW-Lw (key: djrm) 

2. **Data generation** Run `ZLFSR_real/matlab/PrepareRealData.m` to generate training data.

3. **Training** Run `ZLFSR_real/train.py` to perform network training.

4. **Test** Run `ZLFSR_real/test.py` to perform network testing.


## Citation
If you use this code for your research, please cite our paper.


## Contact
If you have any questions about the code, please contact to cyy941027@126.com.
