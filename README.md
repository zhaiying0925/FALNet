# Enhancing Polyp Segmentation Accuracy through Frequency-Aware Augmentation and Ambiguity-Aware Learning

## Prerequisites
- `Python 3.8.0`
- `Pytorch 1.10.1`

This code has been tested using `Pytorch` on a RTX3090 GPU.

## Dataset

Our experiments are conducted on five public polyp datasets: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, and ETIS-LaribPolypDB. To maintain consistency with the experimental setup, we follow the data partitioning scheme of the PraNet benchmark: 900 images from Kvasir-SEG and 550 images from CVC-ClinicDB are used as the training set, while the remaining samples along with the other three complete datasets (CVC-ColonDB, CVC-300, and ETIS-LaribPolypDB) serve as the test set.

## Training and Testing
'''
# Training
python train.py
# Testing
python inference.py
'''

