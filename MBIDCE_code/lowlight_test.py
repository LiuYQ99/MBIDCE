import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from torchstat import stat

 
def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    # data_lowlight = Image.open(image_path).resize((725,480))
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    # DCE_net = model.enhance_net_nopool()
    # stat(DCE_net, (3, 1200, 900))
    DCE_net.load_state_dict(torch.load('snapshots1/Epoch99.pth'))
    start = time.time()
    enhanced_image,_ = DCE_net(data_lowlight)
    # _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('test_data','result')
    result_path = image_path

    torchvision.utils.save_image(enhanced_image, result_path)
    # torchvision.utils.save_image(x_1, result_path1)
    # torchvision.utils.save_image(x_2, result_path2)

if __name__ == '__main__':
# test_images
    with torch.no_grad():
        filePath = 'data/test_data/'
	
        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath+file_name+"/*")
            for image in test_list:
                image = image.replace('\\','/')
                print(image)
                lowlight(image)

		

