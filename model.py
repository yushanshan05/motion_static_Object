#coding=utf-8
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torch.nn.functional as F
import time
import os
import copy
import cv2
from PIL import Image
import shutil

from python.getBBoxGroup import getBBoxGroup, drawBBoxGroup
from python.jitterCorrection import jitterCorrection

isDebug=False #False  #True
PRINT = True #True
LABEL=["motion","static"]
input_size=[16,16]
GPU=False

import logging
model_logger = logging.getLogger('motionstatic.model')

class Model():
    
    def __init__(self,weights_path,iou_th):
    
        self.iou_th = iou_th    
        self.input = 16
        self.num_classes = 2
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if GPU:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.model = models.video.r2plus1d_18(pretrained=False, progress=True)              
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes) 
        
        self.model = self.model.to(self.device)

        self.model.eval() 
        
        if GPU:
            checkpoint=torch.load(weights_path, map_location='cuda:0')   #使用GPU
        else:
            checkpoint=torch.load(weights_path, map_location='cpu')   #使用CPU
        #checkpoint=torch.load(weights_path, map_location='cuda:0')
        print("load weights from {}".format(weights_path))    
        self.model.load_state_dict(checkpoint['model_net'])  
        
        self.transform=transforms.Compose([
        transforms.Resize(input_size),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216,0.394666,0.37645],
                             std=[0.22803,0.22145,0.216989])
        ])
        
        print ("model is ok")
        
    def judgeImgSizeIsOK(self,img_lst): #判断一组图片大小是否一致
        num = len(img_lst)
        w = 0
        h = 0
        isOK = True
        if num > 0:
            w = img_lst[0].shape[1]
            h = img_lst[0].shape[0]
            
        for i in range(1,num):
            w_temp = img_lst[i].shape[1]
            h_temp = img_lst[i].shape[0]
            if w_temp!=w or h_temp!=h:
                isOK = False
                
        return isOK

    def predictOneGroupObject(self,img_tensor):
    
        self.model.eval()
        
        img_tensor = img_tensor.to(self.device)
        
        #print("inputs:", img.size())
        
        start_model_time=time.time()
        outputs = self.model(img_tensor) 
        end_model_time=time.time()
        
        outputs = F.softmax(outputs,dim=1) 
        scores, preds = torch.max(outputs, 1) 
        
        label_id_lst=[]        
        score_lst=[]
        for j in range(preds.size()[0]):
            label_id=preds[j].item()
            label_id_lst.append(label_id)
            score_lst.append(scores[j].item())
            
        
        
        #0-->MOTION
        '''label="motion"
        score=scores.item()
        label_id=preds.item()
        if label_id==0:
            label=LABEL[0]            
        else:
            label=LABEL[1]
            
        print("{}:{:.2f}".format(label, score))
            
        print("inputs:", img.size())
        print("outputs:", outputs.size())
        print("outputs[0]:", outputs[0][0].item())
        print("outputs[1]:", outputs[0][1].item())
        print("preds:", preds.item())
        print("scores:", scores.item())
        
        predict_rst=[label_id,score]'''
        #print("label_id_lst:", label_id_lst)
        #print("score_lst:", score_lst)
        return label_id_lst, score_lst  
    
        
    def judgeObjectWork(self,img_all,img_all_bgr,cls_lst_all,score_lst_all,x1_lst,y1_lst,x2_lst,y2_lst,iou_th):       
        
        if PRINT:
            print("cls_lst_all:", cls_lst_all)
            print("score_lst_all:", score_lst_all)
            print("x1_lst:", x1_lst)
            print("y1_lst:", y1_lst)
            print("x2_lst:", x2_lst)
            print("y2_lst:", y2_lst)
        ########## 加校正 ##################################################
        start_tmp = time.time()
        labelInfo_lst=[[60,55,900,140], [1340,955,1920,1030]]
        jitterAlgorithmName="calcOffsetByECC" #calcOffsetByECC  calcOffsetByORB  calcOffsetByOpticalLK
        
        debugPath=""
        if isDebug:
            debugPath = "./debug_info/" 
            if not os.path.exists(debugPath):
                os.makedirs(debugPath)
            
        xmin_lst_all_tmp,ymin_lst_all_tmp,xmax_lst_all_tmp,ymax_lst_all_tmp = jitterCorrection(img_all_bgr, x1_lst,y1_lst,x2_lst,y2_lst, jitterAlgorithmName, labelInfo_lst, isDebug,debugPath) #[[60,55,900,140], [1340,955,1920,1030]]
        jitterCorrection_time=time.time() - start_tmp
        print (jitterCorrection_time)
        model_logger.info("jitterCorrection_time:{}".format(jitterCorrection_time))
        
        ########## 分组 ##################################################
        start_tmp = time.time()
        xmin_lst_all = copy.deepcopy(xmin_lst_all_tmp)
        ymin_lst_all = copy.deepcopy(ymin_lst_all_tmp)
        xmax_lst_all = copy.deepcopy(xmax_lst_all_tmp)
        ymax_lst_all = copy.deepcopy(ymax_lst_all_tmp)        
        
        bbox_lst = getBBoxGroup(img_all_bgr, xmin_lst_all, ymin_lst_all, xmax_lst_all, ymax_lst_all, cls_lst_all, iou_th, isDebug, debugPath)
        if isDebug:
            drawBBoxGroup(img_all_bgr, bbox_lst, debugPath)
        
        getBBoxGroup_time=time.time() - start_tmp
        print (getBBoxGroup_time)
        model_logger.info("getBBoxGroup_time:{}".format(getBBoxGroup_time))
        
        ########## 预测 ##################################################
        start_tmp = time.time()
        expend = 0.4
        group_bbox_list_rst = self.predict_all_object(bbox_lst,img_all,expend,debugPath)
        predict_all_object_time=time.time() - start_tmp
        print (predict_all_object_time)
        model_logger.info("predict_all_object_time:{}".format(predict_all_object_time))
        
        
        return group_bbox_list_rst 
        #return bbox_lst #c_lst_res,s_lst_res,x1_lst_res,y1_lst_res,x2_lst_res,y2_lst_res    
    
        
    def predict_all_object(self,group_bbox_list,group_img_list,expend,debugPath):
    
        group_bbox_list_rst=[]
        
        #print("!!!!!!!!!!!!!!!!!")
        if PRINT:
            print("group_bbox_list info:", group_bbox_list)
            print("group_bbox_list len:", len(group_bbox_list))
        model_logger.info("group_bbox_list:{}".format(group_bbox_list))
        
        
        for j in range(len(group_bbox_list)):  
            img_crop_tensor = torch.empty(0, 3, 2, 16, 16)
            if PRINT:
                print(group_bbox_list[j])
            
            
            #一组数据中有效张数必须大于2张的标志
            valid_cnt=0
            save_cnt=0
            for i in range(len(group_img_list)):                
                if group_bbox_list[j][i][5] == 1:
                    valid_cnt = valid_cnt+1   


            #如果超过两张图，如果只有有效的两帧，这两帧不能是首尾的同一张图，否组数据也无效
            if len(group_img_list)>2:
                if valid_cnt == 2:
                    first_falg=False
                    last_flag=False
                    for i in range(len(group_img_list)):                
                        if group_bbox_list[j][i][5] == 1:
                            if i==0:
                                first_falg = True
                            if i== len(group_img_list)-1:
                                last_flag = True
                    if first_falg and last_flag:
                       valid_cnt=1
                    
            if valid_cnt >1:
            
                group_bbox_list_rst.append([]) 
                #print(group_bbox_list_rst)
               
            
                for i in range(len(group_img_list)-1):
        
                    img_orig0 = group_img_list[i] #cv2.imread(group_img_path_list[i])
                    img_orig1 = group_img_list[i+1] #cv2.imread(group_img_path_list[i+1])

                    #for j in range(len(group_bbox_list)):
                    #print(j)
                    
                    #print (group_bbox_list[j])
                    
                    cls = group_bbox_list[j][i][4] 
                    #print (cls)                
                    
                    xmin0 = group_bbox_list[j][i][0]
                    ymin0 = group_bbox_list[j][i][1]
                    xmax0 = group_bbox_list[j][i][2]
                    ymax0 = group_bbox_list[j][i][3]
                    
                    xmin1 = group_bbox_list[j][i+1][0]
                    ymin1 = group_bbox_list[j][i+1][1]
                    xmax1 = group_bbox_list[j][i+1][2]
                    ymax1 = group_bbox_list[j][i+1][3]
                    
                    '''if group_bbox_list[j][i][5] == 0 and group_bbox_list[j][i+1][5] == 0:
                        group_bbox_list_rst[-1].extend([])'''
                    
                    #如果len(group_img_path_list)-1==1,不可能有构造的0的box
                    if group_bbox_list[j][i][5] == 0:
                        group_bbox_list_rst[-1].append([])
                        continue
                    else:
                        #crop
                        offset_w0 = int(abs(xmax0 - xmin0) * expend)
                        offset_h0 = int(abs(xmax0 - xmin0) * expend)
                        offset_w1 = int(abs(xmax1 - xmin1) * expend)
                        offset_h1 = int(abs(xmax1 - xmin1) * expend)
                        img_crop0 = img_orig0[int(max(ymin0-offset_h0,1)):int(min(ymax0+offset_h0,1080)), int(max(xmin0-offset_w0,1)):int(min(xmax0+offset_w0,1920))]
                        img_crop1 = img_orig1[int(max(ymin1-offset_h1,1)):int(min(ymax1+offset_h1,1080)), int(max(xmin1-offset_w1,1)):int(min(xmax1+offset_w1,1920))]
                        
                        if img_crop0.shape[0]<5 or img_crop0.shape[1]<5 or img_crop1.shape[0]<5 or img_crop1.shape[1]<5:
                            group_bbox_list_rst[-1].append([])
                            continue                            
                        
                        if isDebug:
                            save_cnt=save_cnt+1
                            save_dir=os.path.join(debugPath,str(j)+'_'+str(save_cnt))
                            if os.path.exists(save_dir):
                                shutil.rmtree(save_dir)
                            os.makedirs(save_dir)
                            
                            img_crop0_bgr = cv2.cvtColor(img_crop0, cv2.COLOR_RGB2BGR)
                            save0_path=os.path.join(save_dir,"0.jpg")
                            cv2.imwrite(save0_path, img_crop0_bgr)
                            
                            img_crop1_bgr = cv2.cvtColor(img_crop1, cv2.COLOR_RGB2BGR)
                            save1_path=os.path.join(save_dir,"1.jpg")
                            cv2.imwrite(save1_path, img_crop1_bgr)
                            
                                                        
                        img1=Image.fromarray(img_crop0)                
                        img1=self.transform(img1)                
                        img2=Image.fromarray(img_crop1)                 
                        img2=self.transform(img2)                             
                        
                        images=[]
                        images.append(np.array(img1)) 
                        images.append(np.array(img2))
                        images_np=np.array(images)
                        
                        # swap dimensions to [T,C,H,W]->[C, T H, W]
                        images = torch.from_numpy(images_np).permute(1,0,2,3)
                        #print("images shape:", images.size())
                        img = images.unsqueeze(0)
                        
                        img_crop_tensor = torch.cat([img_crop_tensor, img], 0)
                        
                        #如果只有两张图，特殊处理
                        if len(group_img_list)-1==1:
                            group_bbox_list_rst[-1].append([xmin0,ymin0,xmax0,ymax0, cls])
                            group_bbox_list_rst[-1].append([xmin1,ymin1,xmax1,ymax1, cls])
                        else:  
                            group_bbox_list_rst[-1].append([xmin0,ymin0,xmax0,ymax0, cls])
                        #print(i)
                        #print(group_bbox_list_rst)
                        
                    
                label_id_lst, score_lst=self.predictOneGroupObject(img_crop_tensor)
                
                static_cnt = 0
                motion_cnt = 0
                for index in range(len(label_id_lst)):
                    
                    if label_id_lst[index]==0:
                        motion_cnt = motion_cnt +1
                if PRINT:
                    print("all group info before del:", group_bbox_list_rst)
                    print("model predict:{}".format(label_id_lst))
                    
                model_logger.info("all group info before del:{}".format(group_bbox_list_rst))
                model_logger.info("model predict:{}".format(label_id_lst))
                #同一组没有运动的目标则删掉
                if motion_cnt == 0:
                    del group_bbox_list_rst[-1]
                else: 
                    score_index=0 
                    if PRINT:
                        print("now group info:", group_bbox_list_rst[-1])
                    model_logger.info("now group info:{}".format(group_bbox_list_rst[-1]))
                    for iii in group_bbox_list_rst[-1]:
                        '''print("iii:", iii)
                        print("score_index:", score_index)
                        print("score_lst:", score_lst)'''
                        
                        if len(iii)!=0:
                            #如果只有两张图，特殊处理，此时只有一个分数
                            if len(group_img_list)-1==1:                                
                                iii.append(score_lst[0])
                            else:                        
                                iii.append(score_lst[score_index])
                                score_index = score_index+1   #这里如果是原来是静止，修改为运动后，仍用静止的分数
            if PRINT:
                print("all group info after del:",group_bbox_list_rst)
            model_logger.info("all group info after del:{}".format(group_bbox_list_rst))
            
        if PRINT:
            print(group_bbox_list_rst)    
        return  group_bbox_list_rst          
                
                
            
                
               
                
                
            
            
        
    