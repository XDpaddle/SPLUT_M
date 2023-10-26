#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.autograd import Variable
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob


from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter

EXP_NAME = "SP-LUT"

UPSCALE = 4     # upscaling factor

NB_BATCH = 16        # mini-batch
CROP_SIZE = 48       # input LR training patch size

START_ITER = 18000      # Set 0 for from scratch, else will load saved params and trains further
NB_ITER = 200000    # Total number of training iterations

I_DISPLAY = 200     # display info every N iteration
I_VALIDATION = 5000  # validate every N iteration
I_SAVE = 5000       # save models every N iteration

TRAIN_DIR = 'E:/datasets/DIV2K/DIV2K/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = 'E:/PycharmCode/SPLUT/training_testing_code/test_dataset/Set5/'      # Validation images

LR_G = 1e-3         # Learning rate for the generator

QQ=2**4
OUT_NUM=4
VERSION = "SPLUT_M_{}_{}".format(LR_G,OUT_NUM)

### Tensorboard for monitoring ###
writer = SummaryWriter(log_dir='./log/{}'.format(str(VERSION)))

class _baseq(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, steps):
        y_step_ind=paddle.floor(x / steps)
        y = y_step_ind * steps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, None
        return grad_output

class BASEQ(nn.Layer):
    def __init__(self, lvls,activation_range):#lvis=16,activation_range=8
        super(BASEQ, self).__init__()
        self.lvls = lvls
        self.activation_range = activation_range    #8.0
        self.steps = 2 * activation_range / self.lvls   #1

    def forward(self, x):
        x=(((-x - self.activation_range).abs() - (x - self.activation_range).abs()))/2.0
        x[x > self.activation_range-0.1*self.steps] =self.activation_range-0.1*self.steps
        return _baseq.apply(x, self.steps)

### A lightweight deep network ###
class SRNet(nn.Layer):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale
        self.lvls = 16
        self.quant= BASEQ(self.lvls ,8.0)
        self.out_channal= OUT_NUM   #4

        #cwh
        self.lut122=nn.Sequential(
            nn.Conv2D(1, 64, [2,2], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 8, 1, stride=1, padding=0, dilation=1)
        )

        self.lut221=nn.Sequential(
            nn.Conv2D(2,  64, [2,1], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 8, 1, stride=1, padding=0, dilation=1)
        )

        self.lut212=nn.Sequential(
            nn.Conv2D(2,  64, [1,2], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 8, 1, stride=1, padding=0, dilation=1)
        )

        self.lut221_c12=nn.Sequential(
            nn.Conv2D(2,  64, [2,1], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 16, 1, stride=1, padding=0, dilation=1)
        )

        self.lut212_c34=nn.Sequential(
            nn.Conv2D(2,  64, [1,2], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2D(64, 16, 1, stride=1, padding=0, dilation=1)
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        '''
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)'''

    def forward(self, x_in):
        B, C, H, W = x_in.shape
        x_in = x_in.reshape([B*C, 1, H, W])
        x = self.lut122(x_in)+x_in[:,:,:H-1,:W-1]
        x_temp=x_in
        x221 = self.lut221(self.quant(F.pad(x[:,0:2,:,:],(0,0,0,1), mode='reflect')+F.pad(x[:,2:4,:,:],(0,0,1,0), mode='reflect')))
        x212 = self.lut212(self.quant(F.pad(x[:,4:6,:,:],(0,1,0,0), mode='reflect')+F.pad(x[:,6: ,:,:],(1,0,0,0), mode='reflect')))
        x=(x221+x212)/2.0+x

        x2 = self.lut221_c12(self.quant(F.pad(x[:,0:2,:,:],(0,0,0,1), mode='reflect')+F.pad(x[:,2:4,:,:],(0,0,1,0), mode='reflect')))
        x3 = self.lut212_c34(self.quant(F.pad(x[:,4:6,:,:],(0,1,0,0), mode='reflect')+F.pad(x[:,6: ,:,:],(1,0,0,0), mode='reflect')))
        x=(x2+x3)/2.0+x_temp[:,:,:H-1,:W-1]
        x = self.pixel_shuffle(x)
        x = x.reshape([B, C, self.upscale*(H-1), self.upscale*(W-1)])
        return x

if __name__ == '__main__':

    model_G_A = SRNet(upscale=UPSCALE)
    model_G_B = SRNet(upscale=UPSCALE)

    ## Optimizers
    #params_G_A = list(filter(lambda p: p.requires_grad, model_G_A.parameters()))
    #params_G_B = list(filter(lambda p: p.requires_grad, model_G_B.parameters()))
    params_G_A=list(model_G_A.parameters())
    params_G_B=list(model_G_B.parameters())
    opt_G = optim.Adam(learning_rate=LR_G,parameters=[{'params':params_G_A},{'params':params_G_B}])

    ## Load saved params
    if START_ITER > 0:
        lm = paddle.load('checkpoint/{}/model_G_A_i{:06d}.pdparams'.format(str(VERSION), START_ITER))
        model_G_A.set_state_dict(lm)

        lm = paddle.load('checkpoint/{}/model_G_B_i{:06d}.pdparams'.format(str(VERSION), START_ITER))
        model_G_B.set_state_dict(lm)

        lm = paddle.load('checkpoint/{}/opt_G_i{:06d}.pdopt'.format(str(VERSION), START_ITER))
        opt_G.set_state_dict(lm)

    # Training dataset
    Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K( 
                                    datadir = TRAIN_DIR,
                                    crop_size = CROP_SIZE, 
                                    crop_per_image = NB_BATCH//4,
                                    out_batch_size = NB_BATCH,
                                    scale_factor = UPSCALE,
                                    shuffle=True),)
    Iter_H.start(max_q_size=16, workers=4)

    ## Prepare directories
    if not isdir('checkpoint'):
        mkdir('checkpoint')
    if not isdir('result'):
        mkdir('result')
    if not isdir('checkpoint/{}'.format(str(VERSION))):
        mkdir('checkpoint/{}'.format(str(VERSION)))
    if not isdir('result/{}'.format(str(VERSION))):
        mkdir('result/{}'.format(str(VERSION)))

    ## Some preparations 
    print('===> Training start')
    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    def SaveCheckpoint(i, best=False):
        str_best = ''
        if best:
            str_best = '_best'

        paddle.save(model_G_A.state_dict(), 'checkpoint/{}/model_G_A_i{:06d}{}.pdparams'.format(str(VERSION), i, str_best ))
        paddle.save(model_G_B.state_dict(), 'checkpoint/{}/model_G_B_i{:06d}{}.pdparams'.format(str(VERSION), i, str_best ))
        paddle.save(opt_G.state_dict(), 'checkpoint/{}/opt_G_i{:06d}{}.pdopt'.format(str(VERSION), i, str_best))
        print("Checkpoint saved")

    def SpaceInference(model,batch):    #batch:torch.Size([32, 3, 48, 48])
        batch_S1 = model(F.pad(batch, (0,1,0,1), mode='reflect'))
        batch_S = paddle.clip(batch_S1,-1,1)
        return batch_S

    ### TRAINING
    for i in tqdm(range(START_ITER+1, NB_ITER+1)):

        model_G_A.train()
        model_G_B.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = Iter_H.dequeue()
        
        batch_H = paddle.to_tensor(batch_H)      # BxCxHxW, range [0,1] [32,3,192,192]
        batch_L = paddle.to_tensor(batch_L)     # BxCxHxW, range [0,1] [32,3,48,48]

        #[0,1]-->[0,255]
        batch_L225 = paddle.floor(batch_L*255)
        #A=paddle.to_tensor(QQ,dtype='int64')
        batch_L_A= paddle.floor_divide(paddle.to_tensor(batch_L225,dtype='int64'), paddle.to_tensor(QQ,dtype='int64'))  #QQ=16  MSB
        #batch_L_A = batch_L225 // QQ
        batch_L_B= batch_L225 % QQ
        batch_L_A=batch_L_A/QQ
        batch_L_B=batch_L_B/QQ  #归一化 [0,1]

        dT += time.time() - st

        ## TRAIN G
        st = time.time()
        opt_G.clear_grad()

        batch_S_A=SpaceInference(model_G_A,batch_L_A)
        batch_S_B=SpaceInference(model_G_B,batch_L_B)
        batch_S_all = paddle.clip(batch_S_A+batch_S_B,0,1)

        loss_Pixel = paddle.mean( ((batch_S_all - batch_H)**2))
        loss_G = loss_Pixel

        # Update
        loss_G.backward()
        opt_G.step()
        rT += time.time() - st

        # For monitoring
        accum_samples += NB_BATCH
        l_accum[0] += loss_Pixel.item()

        ## Show information
        if i % I_DISPLAY == 0:
            writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)
            print("{} {}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.

        ## Save models
        if i % I_SAVE == 0:
            SaveCheckpoint(i)

        ## Validation
        if i % I_VALIDATION == 0:
            with paddle.no_grad():
                model_G_A.eval()
                model_G_B.eval()

                # Test for validation images
                files_gt = glob.glob(VAL_DIR + '/HR/*.png')
                files_gt.sort()
                files_lr = glob.glob(VAL_DIR + '/LR/*.png')
                files_lr.sort()

                psnrs = []
                lpips = []

                for ti, fn in enumerate(files_gt):
                    # Load HR image
                    tmp = _load_img_array(files_gt[ti])
                    val_H = np.asarray(tmp).astype(np.float32)  # HxWxC

                    # Load LR image
                    tmp = _load_img_array(files_lr[ti])
                    val_L = np.asarray(tmp).astype(np.float32)  # HxWxC
                    val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
                    val_L = val_L[np.newaxis, ...]            # BxCxHxW

                    val_L = paddle.to_tensor(val_L.copy())
                    #pdb.set_trace()
                    val_L = paddle.floor(val_L*255)
                    #val_L_A= val_L// QQ
                    val_L_A = paddle.floor_divide(paddle.to_tensor(val_L, dtype='int64'),
                                                    paddle.to_tensor(QQ, dtype='int64'))
                    val_L_B= val_L % QQ
                    val_L_A=val_L_A/QQ
                    val_L_B=val_L_B/QQ

                    batch_S_A=SpaceInference(model_G_A,val_L_A)
                    batch_S_B=SpaceInference(model_G_B,val_L_B)
                    batch_S = batch_S_A+batch_S_B

                    # Output 
                    image_out = (batch_S).numpy()
                    image_out = np.transpose(image_out[0], [1, 2, 0])  # HxWxC
                    image_out = np.clip(image_out, 0. , 1.)      # CxHxW
                    
                    # Save to file
                    image_out = ((image_out)*255).astype(np.uint8)
                    # Image.fromarray(image_out).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))

                    # PSNR on Y channel
                    img_gt = (val_H*255).astype(np.uint8)
                    CROP_S = 4
                    psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S))

            print('AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs))))

            writer.add_scalar('PSNR_valid', np.mean(np.asarray(psnrs)), i)
            writer.flush()
