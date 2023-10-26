from os import mkdir
from os.path import isdir
#import torch
#import torch.nn.functional as F
import numpy as np
import paddle
import paddle.nn.functional as F
from Train_SPLUT_M_Paddle import _baseq,BASEQ,SRNet,UPSCALE,VERSION

MODEL_PATH = "./checkpoint/{}".format(VERSION)   # Trained SR net params
ITER=14000

LUT_NUM_LIST=[1,2,2,3,3]
KERNAL_LIST=[122,221,212,221,212]
img_bits=4


class SRNet_LUT(SRNet):
    def __init__(self, upscale=4):
        super(SRNet_LUT, self).__init__()

    def forward(self, x_in):
        if LUT_NUM==1:
            x_out = self.lut122(x_in)
        elif LUT_NUM==2 and KERNAL==221:
            x_out = self.lut221(x_in)
        elif LUT_NUM==2 and KERNAL==212:
            x_out = self.lut212(x_in)
        elif LUT_NUM==3 and KERNAL==221:
            x_out = self.lut221_c12(x_in)
        elif LUT_NUM==3 and KERNAL==212:
            x_out = self.lut212_c34(x_in)
        return x_out

if __name__ == '__main__':

    model_G_A = SRNet_LUT(upscale=UPSCALE)
    model_G_B = SRNet_LUT(upscale=UPSCALE)
    lm = paddle.load('{}/model_G_A_i{:06d}.pdparams'.format(MODEL_PATH, ITER))
    model_G_A.set_state_dict(lm)
    lm = paddle.load('{}/model_G_B_i{:06d}.pdparams'.format(MODEL_PATH, ITER))
    model_G_B.set_state_dict(lm)

    ### Extract input-output pairs
    with paddle.no_grad():
        model_G_A.eval()
        model_G_B.eval()
        for ind in range(len(LUT_NUM_LIST)):    #LUT_NUM_LIST=[1,2,2,3,3]
            LUT_NUM=LUT_NUM_LIST[ind]
            KERNAL=KERNAL_LIST[ind] #KERNAL_LIST=[122,221,212,221,212]

            if LUT_NUM==1:
                L = 2 ** img_bits#16
                base_step_ind=paddle.arange(0, L, 1)#tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
                base=base_step_ind/16.0
                index_4D=paddle.meshgrid(base,base,base,base)
                onebyfourth=paddle.concat([index_4D[0].flatten().unsqueeze(1),index_4D[1].flatten().unsqueeze(1),index_4D[2].flatten().unsqueeze(1),index_4D[3].flatten().unsqueeze(1)],1)
                #onebyfourth:[65536,4]
                base_B=base_step_ind/16.0
                index_4D_B=paddle.meshgrid(base_B,base_B,base_B,base_B)
                onebyfourth_B=paddle.concat([index_4D_B[0].flatten().unsqueeze(1),index_4D_B[1].flatten().unsqueeze(1),index_4D_B[2].flatten().unsqueeze(1),index_4D_B[3].flatten().unsqueeze(1)],1)
                #onebyfourth_B:[65536,4
            else:

                a_rang = 8.0
                L = model_G_A.lvls  #16
                base_steps=2* a_rang/L  #1
                base_step_ind=paddle.arange(0, L, 1)-0.5*L#tensor([-8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,
         #6.,  7.])
                base=base_steps*base_step_ind
                index_4D=paddle.meshgrid(base,base,base,base)
                onebyfourth=paddle.concat([index_4D[0].flatten().unsqueeze(1),index_4D[1].flatten().unsqueeze(1),index_4D[2].flatten().unsqueeze(1),index_4D[3].flatten().unsqueeze(1)],1)
                #onebyfourth:[65536,4
            if LUT_NUM==1:
                input_tensor   = onebyfourth.unsqueeze(1).unsqueeze(1).reshape([-1,1,2,2])#torch.Size([65536, 1, 2, 2])
                input_tensor_B = onebyfourth_B.unsqueeze(1).unsqueeze(1).reshape([-1,1,2,2])#torch.Size([65536, 1, 2, 2])
            elif KERNAL==221:
                input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape([-1,2,2,1])#[65536,2,2,1]
                input_tensor_B = onebyfourth.unsqueeze(1).unsqueeze(1).reshape([-1,2,2,1])#torch.Size([65536, 2, 2, 1])
            elif KERNAL==212:
                input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape([-1,2,1,2])#torch.Size([65536, 2, 1, 2])
                input_tensor_B = onebyfourth.unsqueeze(1).unsqueeze(1).reshape([-1,2,1,2])
            print("Input size: ", input_tensor.shape)

            # Split input to not over GPU memory
            B = input_tensor.shape[0] // 100#655
            outputs_A = []
            outputs_B = []

            for b in range(100):
                if b == 99:
                    batch_output_A = model_G_A(input_tensor[b*B:])
                    batch_output_B = model_G_B(input_tensor_B[b*B:])
                else:
                    batch_output_A = model_G_A(input_tensor[b*B:(b+1)*B])
                    batch_output_B = model_G_B(input_tensor_B[b*B:(b+1)*B])

                outputs_A += [ batch_output_A ]
                outputs_B += [ batch_output_B ]
            
            results_A = np.concatenate(outputs_A, 0)#65536,8,1,1)
            results_B = np.concatenate(outputs_B, 0)
            print("Resulting LUT size: ", results_A.shape)
            if not isdir('./transfer/{}'.format(str(VERSION))):
                mkdir('./transfer/{}'.format(str(VERSION)))
            np.save("./transfer/{}/LUT{}_K{}_Model_S_A".format(VERSION,LUT_NUM,KERNAL), results_A)#"SPLUT_M_{}_{}".format(LR_G,OUT_NUM)
            np.save("./transfer/{}/LUT{}_K{}_Model_S_B".format(VERSION,LUT_NUM,KERNAL), results_B)
            #resultsA,resultsB:float32
            '''
            lut1:(65536,8,1,1) (-11,12)
            lut2:65536,8,1,1)  (-35,30)
            lut3:(65536,8,1,1)
            '''