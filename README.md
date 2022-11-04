# MBIDCE
Multi-branch illumination depth curve enhancemnet

**The implementation of MBIDCE is for non-commercial use only.**

## Requirements
1. Python 3.7 
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0

**MBIDCE does not need special configurations. Just basic environment.** 

**Or you can create a conda environment to run our code like this**:

conda create --name MBIDCE_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch

### Folder structure
Download the MBIDCE_code first.

**Please create folders in data:test_data train_data result**

The following shows the basic folder structure.
```

├── data
│   ├── test_data # testing data. 
│   │   ├── LIME 
│   │   └── MEF
│   │   └── NPE
│   ├──── train_data
│   └──── result  
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # MBIDCE network
├── dataloader.py
├── snapshots
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```

### Test: 

cd MBIDCE_code
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder. You can find the enhanced images in the "result" folder.

### Train: 
**Training data is as the same as Zero-DCE**

1) cd MBIDCE_code

2) download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA">baidu cloud [password: 1234]</a>

3) unzip and put the  downloaded "train_data" folder to "data" folder
```
python lowlight_train.py 
```

##  License
The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International L

## Contact
If you have any questions, please contact Yingqun Liu at yingqunliu9@gmail.com.
