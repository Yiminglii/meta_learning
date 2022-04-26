#meta pre generater 
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class meta_generater(nn.Module):
    def __init__(self,
        patch_size=8,
        pooling_size=2, #定义每次下采样的池化size
        in_channel_nums=3,
    ):
        super(meta_generater,self).__init__()
        self.patch_s=patch_size
        self.num_d=int(math.log(patch_size,(2))) # downsample 下采样次数

        self.ups_layer=nn.Sequential(
            nn.Conv2d(in_channel_nums,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1)
        )

        self.pooling=nn.Sequential(
            nn.MaxPool2d(pooling_size,pooling_size),
            nn.Conv2d(64,64,3,padding=1)
        )
        
        self.down_layer_shot=nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32,in_channel_nums,3,padding=1),
            nn.Sigmoid()
        )
        self.down_layer_long=nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,16,3,padding=1),
            nn.Sigmoid(),
            nn.Conv2d(16,3,3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x,adv,pgd,shuffle):
        x=self.ups_layer(x)
        for i in range(self.num_d):
            x=self.pooling(x)
        map_shot=self.down_layer_shot(x)
        map_long=self.down_layer_long(x)
        
        zero = torch.zeros_like(map_shot)
        one = torch.ones_like(map_shot)

        map_shot=torch.where(map_shot > 0.5, one, zero)
        map_long=torch.where(map_long > 0.5, one*2,zero)
        map_k=map_shot+map_long #获得各个patch上的操作键值

        # map_z=torch.zeros_like(x)
        # b,c,h,w=map_k.size()
        # for i in range(b):
        #     for j in range(c):
        #         for k in range(h):
        #             for n in range(w):
        #                 k=map_k[i][j][k][n]
        #                 print(k)
        #                 map_z[i,j,k*self.patch_s::(k+1)*self.patch_s,n*self.patch_s::(n+1)*self.patch_s]
        #                 # =torch.ones(self.patch_s,self.patch_s)*k
        
        out=F.interpolate(input=map_k,scale_factor=self.patch_s,mode="nearest")
        out=torch.where(out==1,pgd,out)
        out=torch.where(out==2,adv,out)
        out=torch.where(out==3,shuffle,out)

        return out,map_k
