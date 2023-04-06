################################################################
This Project is a Pytorch implementation of "Zero-shot dense light field reconstruction from heterogeneous imaging using Cycle Consistency"

## Requirements
** Python 3.7
** PyTorch 1.2.0
** CUDA 10.0
** Cudnn
** Matlab 2017b (for training/test data generation and evaluation)


## Usage
1. **Dataset** Please download the dataset from http://lightfields.stanford.edu/mvlf/release/

2. **Data generation** Run `matlab/PrepareData.m` to generate training data.

3. **Training** Run `train.py`  to perform network training.

4. **Test** Run `test.py`  to perform network testing.

5. **Evaluation** Run `matlab/evaluate.m` to generate evaluation results.


## Citation
If you use this code for your research, please cite our paper.


## Contact
If you have any questions about the code, please contact to cyy941027@126.com.
