from turtle import pd
from PIL import Image
import numpy as np
from os import mkdir
from os.path import isdir
from tqdm import tqdm
import glob
from basicsr_utils import tensor2img
from basicsr_metrics import calculate_psnr,calculate_ssim
#import torch
import paddle
from Train_SPLUT_M_Paddle import VERSION

UPSCALE = 4     # upscaling factor
TEST_DIR = './test_dataset/Set5'
# TEST_DIR = './test_dataset/Set14'
# TEST_DIR ='./test_dataset/BSDS100'
# TEST_DIR ='./test_dataset/urban100'
# TEST_DIR ='./test_dataset/manga109'

# Load LUT
LUTA1_122 = np.load("./transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,1,122)).astype(np.float32)#(65536, 8, 1, 1) float32 min-11
LUTA2_221 = np.load("./transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,2,221)).astype(np.float32)#(65536, 8, 1, 1) float32 min -35
LUTA2_212 = np.load("./transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,2,212)).astype(np.float32)#(65536,8,1,1)
LUTA3_221 = np.load("./transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,3,221)).astype(np.float32)#(65536, 16, 1, 1)
LUTA3_212 = np.load("./transfer/{}/LUT{}_K{}_Model_S_A.npy".format(VERSION,3,212)).astype(np.float32)#(65536, 16, 1, 1) float32 min -1

LUTB1_122 = np.load("./transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,1,122)).astype(np.float32)#(65536, 8, 1, 1)
LUTB2_221 = np.load("./transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,2,221)).astype(np.float32)#(65536, 8, 1, 1)
LUTB2_212 = np.load("./transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,2,212)).astype(np.float32)#(65536, 8, 1, 1)
LUTB3_221 = np.load("./transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,3,221)).astype(np.float32)#(65536, 16, 1, 1)
LUTB3_212 = np.load("./transfer/{}/LUT{}_K{}_Model_S_B.npy".format(VERSION,3,212)).astype(np.float32)#(65536, 16, 1, 1)

# Test LR images
files_lr = glob.glob(TEST_DIR + '/LR/*.png'.format(UPSCALE))
files_lr.sort()
# Test GT images
files_gt = glob.glob(TEST_DIR + '/HR/*.png')
files_gt.sort()

L = 16

def LUT1_122(weight, img_in):
    C, H, W = img_in.shape
    img_in = img_in.reshape(C, 1, H, W)#(3,1,129,129

    img_a1 = img_in[:,:, 0:H-1, 0:W-1]#(3,1,128,128)
    img_b1 = img_in[:,:, 0:H-1, 1:W  ]#(3,1,128,128)
    img_c1 = img_in[:,:, 1:H  , 0:W-1]#(3,1,128,128)
    img_d1 = img_in[:,:, 1:H  , 1:W  ]#(3,1,128,128)

    out = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], -1))   
    out = np.transpose(out, (0, 1, 4, 2, 3)).reshape((img_a1.shape[0], -1,img_a1.shape[2],img_a1.shape[3]))#(3, 8, 128, 128)
    return out
#L=16
def LUT23(weight, img_in,a_range,ker,nlut):
    img_in = np.clip(img_in, -a_range, a_range-0.01)
    B ,IC, H, W = img_in.shape
    if ker=='k221':#img_in:(3,2,129,128)
        img_a1 = img_in[:,0:IC-1, 0:H-1,:]+L/2#(3,1,128,128)
        img_b1 = img_in[:,0:IC-1, 1:H  ,:]+L/2#(3,1,128,128)
        img_c1 = img_in[:,1:IC  , 0:H-1,:]+L/2#(3,1,128,128)
        img_d1 = img_in[:,1:IC  , 1:H  ,:]+L/2#(3,1,128,128)
    else:
        img_a1 = img_in[:,0:IC-1, :,0:W-1]+L/2#(3,1,128,128)
        img_b1 = img_in[:,0:IC-1, :,1:W  ]+L/2#(3,1,128,128)
        img_c1 = img_in[:,1:IC  , :,0:W-1]+L/2#(3,1,128,128)
        img_d1 = img_in[:,1:IC  , :,1:W  ]+L/2#(3,1,128,128)  min均为0
    out = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ]#(49152, 16, 1, 1)
    if nlut==2:
        out=out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], -1))  # (3, 1, 128, 128, 8)
        out = np.transpose(out, (0, 1, 4, 2, 3)).reshape((img_a1.shape[0], -1,img_a1.shape[2],img_a1.shape[3]))#(3, 8, 128, 128)
    else:
        out=out.reshape((img_a1.shape[0], img_a1.shape[2], img_a1.shape[3],-1))   #(3, 128, 128, 16)
        out=np.transpose(out, (0, 3, 1, 2))#(3, 16, 128, 128)
    return out

psnrs = []
ssims = []
for ti, fn in enumerate(tqdm(files_gt)):
    # Load LR image
    img_lr = np.array(Image.open(files_lr[ti])).astype(np.float32)#numpy (128,128,3) float32
    h, w, c = img_lr.shape
    # Load GT image
    img_gt = np.array(Image.open(files_gt[ti]))#numpy (512,512,3) uint8

    img_in = np.pad(img_lr, ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))#pad后：（129，129，3）
    img_in_A255 = img_in// L#L=16
    img_in_B255 = img_in % L
    img_in_A = img_in_A255/L
    img_in_B = img_in_B255/L

    # A
    x_layer1=LUT1_122(LUTA1_122,img_in_A255)+img_in_A[:,0:h,0:w].reshape((3,1,h,w))#(3,8,128,128)
    x2_in1=np.pad(x_layer1[:,0:2,:,:],((0,0),(0,0),(0,1),(0,0)), mode='reflect')+np.pad(x_layer1[:,2:4,:,:],((0,0),(0,0),(1,0),(0,0)), mode='reflect')
    x2_in2=np.pad(x_layer1[:,4:6,:,:],((0,0),(0,0),(0,0),(0,1)), mode='reflect')+np.pad(x_layer1[:,6: ,:,:],((0,0),(0,0),(0,0),(1,0)), mode='reflect')
    x_layer2=(LUT23(LUTA2_221, x2_in1,8.0 ,'k221',2)+LUT23(LUTA2_212, x2_in2,8.0 ,'k212',2))/2.0+x_layer1
    x3_in1=np.pad(x_layer2[:,0:2,:,:],((0,0),(0,0),(0,1),(0,0)), mode='reflect')+np.pad(x_layer2[:,2:4,:,:],((0,0),(0,0),(1,0),(0,0)), mode='reflect')
    x3_in2=np.pad(x_layer2[:,4:6,:,:],((0,0),(0,0),(0,0),(0,1)), mode='reflect')+np.pad(x_layer2[:,6: ,:,:],((0,0),(0,0),(0,0),(1,0)), mode='reflect')
    img_out=(LUT23(LUTA3_221, x3_in1,8.0 ,'k221',3)+LUT23(LUTA3_212, x3_in2,8.0 ,'k212',3))/2.0+img_in_A[:,0:h,0:w].reshape((3,1,h,w))
    img_out=img_out.reshape((3,UPSCALE,UPSCALE,h,w))#(3, 4, 4, 128, 128)
    img_out=np.transpose(img_out,(0,3,1,4,2)).reshape((3,UPSCALE*h,UPSCALE*w))   # (3, 512, 512)
    img_out_A = img_out.transpose((1,2,0))#(512, 512, 3)
    # B
    x_layer1=LUT1_122(LUTB1_122,img_in_B255)+img_in_B[:,0:h,0:w].reshape((3,1,h,w))
    x2_in1=np.pad(x_layer1[:,0:2,:,:],((0,0),(0,0),(0,1),(0,0)), mode='reflect')+np.pad(x_layer1[:,2:4,:,:],((0,0),(0,0),(1,0),(0,0)), mode='reflect')
    x2_in2=np.pad(x_layer1[:,4:6,:,:],((0,0),(0,0),(0,0),(0,1)), mode='reflect')+np.pad(x_layer1[:,6: ,:,:],((0,0),(0,0),(0,0),(1,0)), mode='reflect')
    x_layer2=(LUT23(LUTB2_221, x2_in1,8.0 ,'k221',2)+LUT23(LUTB2_212, x2_in2,8.0 ,'k212',2))/2.0+x_layer1
    x3_in1=np.pad(x_layer2[:,0:2,:,:],((0,0),(0,0),(0,1),(0,0)), mode='reflect')+np.pad(x_layer2[:,2:4,:,:],((0,0),(0,0),(1,0),(0,0)), mode='reflect')
    x3_in2=np.pad(x_layer2[:,4:6,:,:],((0,0),(0,0),(0,0),(0,1)), mode='reflect')+np.pad(x_layer2[:,6: ,:,:],((0,0),(0,0),(0,0),(1,0)), mode='reflect')
    img_out=(LUT23(LUTB3_221, x3_in1,8.0 ,'k221',3)+LUT23(LUTB3_212, x3_in2,8.0 ,'k212',3))/2.0+img_in_B[:,0:h,0:w].reshape((3,1,h,w))
    img_out=img_out.reshape((3,UPSCALE,UPSCALE,h,w))
    img_out=np.transpose(img_out,(0,3,1,4,2)).reshape((3,UPSCALE*h,UPSCALE*w))    
    img_out_B = img_out.transpose((1,2,0))#（512，512，3）

    img_out=img_out_A+img_out_B
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    if img_gt.shape[0] > img_out.shape[0]:
        img_out = np.pad(img_out, ((0,img_gt.shape[0]-img_out.shape[0]),(0,0),(0,0)))
    if img_gt.shape[1] > img_out.shape[1]:
        img_out = np.pad(img_out, ((0,0),(0,img_gt.shape[1]-img_out.shape[1]),(0,0)))

    # # Save to file
    if not isdir(TEST_DIR+'/SR'):
        mkdir(TEST_DIR+'/SR')
    Image.fromarray(img_out).save(TEST_DIR+'/SR/{}.png'.format(fn.split('\\')[-1][:-4]))

    img_out = img_out.transpose((2,0,1))
    img_gt  = img_gt.transpose((2,0,1))
    sr_img = tensor2img(paddle.to_tensor(img_out/255.0))
    gt_img = tensor2img(paddle.to_tensor(img_gt/255.0))
    CROP_S = 4
    psnrs.append(calculate_psnr(sr_img, gt_img, CROP_S, 'HWC', True))
    ssims.append(calculate_ssim(sr_img, gt_img, CROP_S, 'HWC', True))

print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))
print('AVG SSIM: {}'.format(np.mean(np.asarray(ssims))))
