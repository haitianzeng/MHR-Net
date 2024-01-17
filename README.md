# MHR-Net
Official PyTorch implementation of ECCV 2022 paper **MHR-Net: Multiple-Hypothesis Reconstruction of Non-Rigid Shapes from 2D Views**. [Paper Link.](https://arxiv.org/abs/2207.09086)

<p align="center"><img src="mhr-net_figure.png" width="100%" alt="" /></p>


## Requirements
* Python 3.7
* PyTorch 1.7.1
* torch-batch-svd
* opencv
* scipy

To setup the environment, we recommend using the following lines to create a new conda env for MHR-Net:
```
conda create -n mhr_net python=3.7
conda activate mhr_net
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install scipy
```
For torch-batch-svd, we use a previous version this library, which can be found in ./torch-batch-svd.zip. Unzip this file and run:
```
python setup.py install
```
in the library directory to install it.

**************************************************************

## Data preparation
We use the data processing from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). The output .npz files should be placed in a data directory like:
```
${PROJECT_ROOT}/
|-- data
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
```
The dataset can also be downloaded from [here](https://drive.google.com/file/d/1EwSIBohZrzbJn-kJqwNvn93JJlHLcesv/view).

***************************************************************

## Training
The general training command is:
```
python train.py -k gt -e [num_epochs] -c [exp_dir] -hp [hparams]
```

We provide the script with the hyper parameters we used in this paper. To achieve a better result, we pre-train the model without procrustean loss for 9 epochs and start full training from the checkpoint. The checkpoint can be downloaded from [here](https://drive.google.com/file/d/1A0mbDJ0bdy0CE1UOkpW20BlmKkfVQ00m/view) and should be put into the checkpoint directory:
```
${PROJECT_ROOT}/
|-- checkpoint
|   |-- run1
|       |-- model_epoch_9.bin
```

After that, run the script to start training:
```
bash ./train_script.sh
```

*******************************************************************

## Acknowledgement
Our code is based on the following repositories. We thank the authors for releasing their codes.

- [ITES](https://github.com/sjtuxcx/ITES)
- [C3DPO](https://github.com/facebookresearch/c3dpo_nrsfm)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)
