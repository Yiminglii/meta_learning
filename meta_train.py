from pickletools import optimize
from pyexpat import model
from meta import meta_generater
import torch
import torch.nn.functional as F
import torch.nn as nn
class meta:
    def __init__(self,
                device,
                tar_model,
                model1,
                model2,
                model3,
                patch_size=8,
                pooling_size=2,
                in_channel_num=3,
                target=-1

    ):
    
        self.device=device
        self.tar=tar_model
        self.model1=model1
        self.model2=model2
        self.model3=model3
        self.target=target

        self.meta_g=meta_generater(patch_size,pooling_size,in_channel_num).to(device)
        self.optimizer=torch.optim.Adam(self.meta_g.parameters(),lr=0.001)
        self.Loss1=nn.MSELoss()
        self.loss2=nn.MSELoss()
    def train(self,dataloader,epochs):
        for epoch in range(1,epochs+1):

            for i,data in enumerate(dataloader):
                images,labels=data
                images=images.to(self.device)
                labels=labels.to(self.device)
                pgd=self.model1(images)*0.1+images
                adv=self.model2(images)*0.1+images
                shuffle=self.model3(images)*0.1+images

                pre,map_k=self.meta_g(images,pgd,adv,shuffle) 
                pre_labels=self.tar(pre) #计算扰动后的图像的分类情况
                pre_labels=torch.argmax(pre_labels,dim=1)
                if self.target==-1:
                  f_labels = labels[torch.randperm(labels.size(0))]
                else:
                  f_labels = torch.ones_like(labels)*self.target
                loss1 = F.mse_loss(pre_labels.float(),labels.float())
                loss2 = F.mse_loss(pre,images)

                L1=0 #正则化（暂时还没编）
                L2=0

                self.optimizer.zero_grad()
                (loss2+loss1).backward()
                self.optimizer.step()
            
            print('Epoch [%d/%d],  Loss1: %.4f, Loss2 %.4f'%(epoch, epochs, loss1.item(),loss2.item()))




