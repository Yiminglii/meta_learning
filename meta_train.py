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

    ):
    
        self.device=device
        self.tar=tar_model
        self.model1=model1
        self.model2=model2
        self.model3=model3

        self.meta_g=meta_generater(patch_size,pooling_size,in_channel_num).to(device)
        self.optimizer=torch.optim.Adam(self.meta_g.parameters(),lr=0.001)
        self.Loss1=nn.MSELoss()
        self.loss2=nn.MSELoss()
    def train(self,dataloader,epoch):
        for epoch in range(1,epoch+1):
            loss=0

            for i,data in enumerate(dataloader):
                images,labels=data
                images=images.to(self.device)
                labels=labels.to(self.device)
                pgd=self.model1(images)
                adv=self.model2(images)
                shuffle=self.model3(images)

                pre,map_k=self.meta_g(images,pgd,adv,shuffle) 
                pre_labels=self.tar(pre) #计算扰动后的图像的分类情况
                pre_labels=torch.argmax(pre_labels,dim=len(pre_labels))
                # #calculate loss
                # logits_model = self.tar(pre)
                # probs_model = F.softmax(logits_model, dim=1)
                # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

                # # C&W loss function
                # real = torch.sum(onehot_labels * probs_model, dim=1)
                # other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
                # zeros = torch.zeros_like(other)
                # loss_adv = torch.max(real - other, zeros)
                # loss1 = torch.sum(loss_adv)
                loss1 = F.mse_loss(pre_labels,labels)
                loss2 = F.mse_loss(pre,images)

                L1=0 #正则化（暂时还没编）
                L2=0

                self.optimizer.zero_grad()
                (loss2+loss1).backward()
                self.optimizer.step()
            
            print('Epoch [%d/%d],  Loss1: %.4f, Loss2 %.4f'%(epoch+1, epoch, loss1.item(),loss2.item()))






