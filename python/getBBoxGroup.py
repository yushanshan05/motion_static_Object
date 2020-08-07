#coding=utf-8

import cv2,os
import numpy as np
import time,copy

'''
input_path = "/data/AI/zhangjing/Detectron/test_xml/test2_zj/734"
imgPath_lst = ["/data/AI/zhangjing/Detectron/test_xml/test2_zj/734/TYFXTJ-011_83_20200627175549856.jpg", "/data/AI/zhangjing/Detectron/test_xml/test2_zj/734/TYFXTJ-011_83_20200627175551905.jpg"]

img_lst_all = []

for i in range(len(imgPath_lst)):
    img = cv2.imread(imgPath_lst[i])
    img_lst_all.append( img )
    
cls_lst_all = [['car','forklift','tricycle'],['car','forklift','tricycle', 'forklift']]
score_lst_all = [[0.7,0.6,0.5],[0.8,0.7,0.6,0.5]]
xmin_lst_all = [[1159,194,98], [1172,180,144,320]]
ymin_lst_all = [[956,644,1008],[957,649,1010,640]]
xmax_lst_all = [[1374,557,283],[1385,423,302,594]]
ymax_lst_all = [[1047,783,1078],[1048,773,1078,767]]'''

def calcIOU(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2): #计算IOU
    xx1 = np.maximum(xmin1, xmin2)
    yy1 = np.maximum(ymin1, ymin2)
    xx2 = np.minimum(xmax1, xmax2)
    yy2 = np.minimum(ymax1, ymax2)
    w = np.maximum(0.0, xx2-xx1+1)
    h = np.maximum(0.0, yy2-yy1+1)
    inter = w * h        
    o = inter / ((xmax1-xmin1+1)*(ymax1-ymin1+1) + (xmax2-xmin2+1)*(ymax2-ymin2+1)- inter)
    return o

def drawBBoxGroup(img_lst_all, bbox_group_img_lst_all, debugPath='./'):
    font= cv2.FONT_HERSHEY_SIMPLEX
    for img_id in range(len(img_lst_all)):
        img = copy.deepcopy(img_lst_all[img_id])
        for bbox_id in range(len(bbox_group_img_lst_all)):
            one_group_bbox_lst = bbox_group_img_lst_all[bbox_id]
        
            save_path_tmp = debugPath + "_" + str(img_id)+"_group.jpg" #os.path.join(save_path,str(img_id)+".jpg")
            
            x1 = int(one_group_bbox_lst[img_id][0])
            y1 = int(one_group_bbox_lst[img_id][1])
            x2 = int(one_group_bbox_lst[img_id][2])
            y2 = int(one_group_bbox_lst[img_id][3])
            cls = one_group_bbox_lst[img_id][4]
            isOK = one_group_bbox_lst[img_id][5]
            
            if isOK:
                cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0))
                cv2.putText(img, cls + str(bbox_id)+"_" +str(isOK), (x1,y1), font, 1,(0,255,0),2)
            else:
                cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255))
                cv2.putText(img, cls + str(bbox_id)+"_" +str(isOK), (x1,y1), font, 1,(0,0,255),2)
            cv2.imwrite(save_path_tmp, img)
    
def getBBoxGroup(img_lst_all, xmin_lst_all, ymin_lst_all, xmax_lst_all, ymax_lst_all, cls_lst_all, iou_th=0.7, isDebug=False, debugPath='./'):
    imgIndex_lst = []
    
    if isDebug:
        fw = open(os.path.join(debugPath,"getBBoxGroup.txt"), "a+")
        
    x1_lst = []
    y1_lst = []
    x2_lst = []
    y2_lst = []
    c_lst = []
    imgIndex_lst = []
    ######## 将bbox坐标拉成一行，imgIndex_lst记录bbox是那张图片的
    for i in range(len(cls_lst_all)): #image num
        for j in range(len(cls_lst_all[i])): #bbox num
            x1_lst.append(xmin_lst_all[i][j])
            y1_lst.append(ymin_lst_all[i][j])
            x2_lst.append(xmax_lst_all[i][j])
            y2_lst.append(ymax_lst_all[i][j])
            #s_lst.append(score_lst_all[i][j])
            c_lst.append(cls_lst_all[i][j])
            imgIndex_lst.append(i)    #存储某个bbox在哪张图的
        
    rows = len(x1_lst)
    iou_arr = np.zeros((rows,rows),dtype=int)
    iou_arr_temp = np.zeros((rows,rows),dtype=float)
    
    bbox_group_lst = []
    for row in range(rows): # all bbox num
        for col in range(rows):  
            if row<col:  # top 
                xmin1 = x1_lst[row]
                ymin1 = y1_lst[row]
                xmax1 = x2_lst[row]
                ymax1 = y2_lst[row]
                
                xmin2 = x1_lst[col]
                ymin2 = y1_lst[col]
                xmax2 = x2_lst[col]
                ymax2 = y2_lst[col]
                o = calcIOU(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2)
                iou_arr_temp[row][col] = o
                if o>iou_th and (imgIndex_lst[row]!=imgIndex_lst[col]) and (c_lst[row]==c_lst[col]):   ########### 0.9
                    iou_arr[row][col] = 1
                    bbox_group_lst.append([row,col])
                    
    if isDebug:
        fw.write("iou:"+str(iou_arr_temp)+ "\n")
        fw.write("iou<th:"+ str(iou_arr)+"\n")
        #print ("iou:", iou_arr_temp)           
        #print ("iou<th:", iou_arr)
    
    #### 处理特殊情况：7个bbox, [0,1,2]为第一帧图的bbox,[3,4,5,6]为第2帧图的bbox，[1,4] [1,6]出现第二帧有2个bbox满足要求，不符合实际情况 (行有重复)
    #### 处理特殊情况： 22个bbox, [0-11]为第一帧图的bbox,[12-21]为第2帧图的bbox, [7,19] [10,19]出现第一帧有2个bbox满足要求，不符合实际情况  (列有重复)
    iou_arr_tmp = copy.deepcopy(iou_arr)
    img_num = len(img_lst_all)
    
    for id in range(img_num):
        id_lst = []
        for row in range(rows):
            if id == imgIndex_lst[row]:
                id_lst.append(row)
                
        if len(id_lst)<2:
            continue
        
        ###############处理行重复 
        for row in range(rows):
            #row_iou = iou_arr_tmp[row][id_lst[0]:id_lst[-1]+1] #id_lst[0],id_lst[-1]
            #print ("row_iou: ",row,  row_iou)
            one_num = 0
            for col in range(id_lst[0],id_lst[-1]+1):
                if 1==iou_arr_tmp[row][col]:
                    one_num = one_num + 1 
                    
            if one_num > 1:
                one_iou_max = 0
                one_iou_id = 0
                for col in range(id_lst[0],id_lst[-1]+1):
                    if iou_arr_temp[row][col] > one_iou_max:
                        one_iou_max = iou_arr_temp[row][col]
                        one_iou_id = col
                for col in range(id_lst[0],id_lst[-1]+1):
                    if col != one_iou_id:
                        iou_arr_tmp[row][col] = 0
        
        ###############处理列重复                 
        for col in range(rows):                
            #print (id_lst[0],id_lst[-1]+1, iou_arr_tmp.shape)
            #col_iou = iou_arr_tmp[col][id_lst[0]:id_lst[-1]+1]  #id_lst[0],id_lst[-1]
            #print ("col_iou: ",col, col_iou.shape, col_iou)
            one_num = 0
            for row in range(id_lst[0],id_lst[-1]+1):
                if 1==iou_arr_tmp[row][col]:
                    one_num = one_num + 1 
                    
            if one_num > 1:
                one_iou_max = 0
                one_iou_id = 0
                for row in range(id_lst[0],id_lst[-1]+1):
                    if iou_arr_temp[row][col] > one_iou_max:
                        one_iou_max = iou_arr_temp[row][col]
                        one_iou_id = row
                for row in range(id_lst[0],id_lst[-1]+1):
                    if row != one_iou_id:
                        iou_arr_tmp[row][col]= 0
    iou_arr = copy.deepcopy(iou_arr_tmp)
    if isDebug:                    
        #print ("iou_arr: ", iou_arr_tmp)
        #print ("new iou_arr: ", iou_arr)
        fw.write("iou_arr: "+str(iou_arr_tmp)+"\n")
        fw.write("new iou_arr: "+str(iou_arr)+ "\n")
    
    ########### 过滤bbox分组 #############################
    bbox_group_lst_tmp = copy.deepcopy(bbox_group_lst)
    for id in range(len(bbox_group_lst_tmp)):
        if len(bbox_group_lst_tmp[id])<2:
            continue
            
        row = bbox_group_lst_tmp[id][0]
        for id2 in range(1,len(bbox_group_lst_tmp[id])):
            col = bbox_group_lst_tmp[id][id2]
            if iou_arr_tmp[row][col]==0:
                bbox_group_lst[id].remove(bbox_group_lst_tmp[id][id2])
                row = col
    if isDebug:
        #print ("new bbox_group_lst: ", bbox_group_lst)
        fw.write("new bbox_group_lst: "+ str(bbox_group_lst)+"\n")
    ##############################################################
    
    iou_arr_index = np.zeros((rows,1),dtype=int)
    for row in range(rows):     # all bbox num
        for col in range(rows):  
            if row<col:         # top 
                if 1==iou_arr[row][col] and (0==iou_arr_index[row][0] or 0==iou_arr_index[col][0]):
                    iou_arr_index[row][0] = row+1
                    iou_arr_index[col][0] = row+1
                    for i in range(rows):
                        if 1==iou_arr[i][col] or 1==iou_arr[row][i] or 1==iou_arr[i][row] or 1==iou_arr[col][i]:
                            iou_arr_index[i][0] =  row+1
                            
    if isDebug:                           
        #print ("iou_arr_index:", iou_arr_index)
        #print ("iou =< th: ", iou_th)
        fw.write("iou_arr_index:"+ str(iou_arr_index)+"\n")
        fw.write("iou =< th: "+str(iou_th)+"\n")
    
    ############################################################
    bbox_lst = []
    ###################  iou < th  #############################
    mv_arr_index = np.zeros((rows,1),dtype=int)
    for id in range(rows):
        if iou_arr_index[id][0]<1:  #处理时序图像中没有交集的bbox的索引
            bboxOK_index = id
            x1 = x1_lst[bboxOK_index]
            y1 = y1_lst[bboxOK_index]
            x2 = x2_lst[bboxOK_index]
            y2 = y2_lst[bboxOK_index]
            if isDebug:
                #print ("imgIndex x1 y1 x2 y2: ", imgIndex_lst[bboxOK_index], x1,y1,x2,y2)
                fw.write("imgIndex x1 y1 x2 y2: "+ str(imgIndex_lst[bboxOK_index])+"," +str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+ "\n")
            
            imgIndex = imgIndex_lst[bboxOK_index]
            for r in range(len(img_lst_all)):
                bbox_lst.append(x1)
                bbox_lst.append(y1)
                bbox_lst.append(x2)
                bbox_lst.append(y2)
                bbox_lst.append(str(c_lst[bboxOK_index]))
                if r == imgIndex:
                    bbox_lst.append(1)
                else:
                    bbox_lst.append(0)
                    
    ###################  处理IOU>th的bbox  #############################
    if isDebug:                           
        #print ("iou > th: ", iou_th) 
        fw.write("iou > th: "+ str(iou_th)+"\n")
        
    lst_temp = []
    for id in range(rows):
        if iou_arr_index[id]>0:
            lst_temp.append(iou_arr_index[id][0])
    lst_temp = list(set(lst_temp))
    lst_temp.sort()
    
    for j in range(len(lst_temp)):
        lst = []                #存放IOU>th的索引
        for id in range(rows):
            if lst_temp[j]==iou_arr_index[id][0]:
                lst.append(id)
                
        if isDebug:
            #print (lst_temp[j], ":", lst)    #4:[0,2,3,4](索引)
            fw.write(str(lst_temp[j])+ ":"+ str(lst)+ "\n")
                
        if len(lst)<2:
            continue
        else:
            imgIndex_lst_tmp = []
            imgIndex_lst_tmp_no = []
            imgIndex_lst_tmp_yes = []
            for id in range(len(img_lst_all)):
                imgIndex_lst_tmp_no.append(id)
                imgIndex_lst_tmp.append(id)
                
            if len(lst)!=len(img_lst_all):
                for iiii in range(0,len(lst)): #[1,2]
                    imgIndex = imgIndex_lst[lst[iiii]]
                    if imgIndex in imgIndex_lst_tmp_no:
                        imgIndex_lst_tmp_no.remove(imgIndex)

                lst_temp_copy = copy.deepcopy(lst)
                for id in range(len(imgIndex_lst_tmp_no)):
                    lst_temp_copy.insert(imgIndex_lst_tmp_no[id], -1)

                for id in range(len(lst_temp_copy)):
                    v = lst_temp_copy[id]
                    if v<0: # v=-1  
                        id_new = -1
                        for id_tmp in range(id-1,-1,-1):
                            if lst_temp_copy[id_tmp]>-1:
                                id_new = id_tmp
                                break
                                
                        if id_new <0:
                            for id_tmp in range(id+1,len(lst_temp_copy)):
                                if lst_temp_copy[id_tmp]>-1:
                                    id_new = id_tmp
                                    break

                        x11 = x1_lst[lst_temp_copy[id_new]]
                        y11 = y1_lst[lst_temp_copy[id_new]]
                        x12 = x2_lst[lst_temp_copy[id_new]]
                        y12 = y2_lst[lst_temp_copy[id_new]]

                        bbox_lst.append(x11)
                        bbox_lst.append(y11)
                        bbox_lst.append(x12)
                        bbox_lst.append(y12)
                        bbox_lst.append(str(c_lst[lst[0]]))
                        bbox_lst.append(0)
                    else:
                        x11 = x1_lst[lst_temp_copy[id]]
                        y11 = y1_lst[lst_temp_copy[id]]
                        x12 = x2_lst[lst_temp_copy[id]]
                        y12 = y2_lst[lst_temp_copy[id]]

                        bbox_lst.append(x11)
                        bbox_lst.append(y11)
                        bbox_lst.append(x12)
                        bbox_lst.append(y12)
                        bbox_lst.append(str(c_lst[lst[0]]))
                        bbox_lst.append(1)
            else:
                for id in range(0,len(lst)):
                    x11 = x1_lst[lst[id]]
                    y11 = y1_lst[lst[id]]
                    x12 = x2_lst[lst[id]]
                    y12 = y2_lst[lst[id]]

                    bbox_lst.append(x11)
                    bbox_lst.append(y11)
                    bbox_lst.append(x12)
                    bbox_lst.append(y12)
                    bbox_lst.append(str(c_lst[lst[0]]))
                    
                    imgIndex = imgIndex_lst[lst[id]]
                    bbox_lst.append(1)
                    
    if isDebug:
        ######('bbox_lst: ', [194, 644, 557, 783, 'forklift', 1, 194, 644, 557, 783, 'forklift', 0, 98, 1008, 283, 1078, 'tricycle', 1, 98, 1008, 283, 1078, 'tricycle', 0, 180, 649, 423, 773, 'forklift', 0, 180, 649, 423, 773, 'forklift', 1, 144, 1010, 302, 1078, 'tricycle', 0, 144, 1010, 302, 1078, 'tricycle', 1, 320, 640, 594, 767, 'forklift', 0, 320, 640, 594, 767, 'forklift', 1, 1159, 956, 1374, 1047, 'car', 1, 1172, 957, 1385, 1048, 'car', 1])
        #print ("bbox_lst: ", bbox_lst)
        #print ("\n")
        fw.write("bbox_lst: "+ str(bbox_lst)+"\n\n")
    
    ############# 整合输出 #########################
    bbox_group_img_lst_all = []
    if len(bbox_lst)%(len(img_lst_all)*6)>0:
        print ("bbox error: ", bbox_lst)
        return bbox_group_img_lst_all 
    
    bbox_info_len = 6
    one_group_len = len(img_lst_all)*bbox_info_len
    group_num = len(bbox_lst)/one_group_len
    
    for id in range(one_group_len, len(bbox_lst)+1, one_group_len):
        one_group_lst = bbox_lst[id-one_group_len:id]
        lst = []
        for img_id in range(1,len(img_lst_all)+1):
            lst.append( one_group_lst[(img_id-1)*bbox_info_len: img_id*bbox_info_len] )
        bbox_group_img_lst_all.append( lst )
        
    '''if isDebug:
        print ("\n")
        print (" result bbox_group_img_lst_all: ", bbox_group_img_lst_all)'''
        
    return bbox_group_img_lst_all   
    
#getBBoxGroup(img_lst_all, xmin_lst_all, ymin_lst_all, xmax_lst_all, ymax_lst_all, cls_lst_all, 0.7, True)                 