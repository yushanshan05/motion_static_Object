import copy
import cv2
import os
from calcOffset import calcOffsetByECC, calcOffsetByORB, calcOffsetByOpticalLK

def protectXY(x,w):
    x_new = x
    if x<1:
        x_new=1
    elif x>w-1:
        x_new=w-1
    return x_new
    
def jitterCorrection(img_lst, xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all, AlgorithmName="calcOffsetByECC", labelInfo_lst=[[60,55,900,140], [1340,955,1920,1030]], isDebug=False,debugPath='./'): #[[60,55,900,140], [1340,955,1920,1030]]    
    
    if len(img_lst)!=len(xmin_lst_all)!=len(ymin_lst_all)!=len(xmax_lst_all)!=len(ymax_lst_all):
        print ("error: The number of images does not match the length of the bbox!")
    
    xmin_lst_all_res = []
    ymin_lst_all_res = []
    xmax_lst_all_res = []
    ymax_lst_all_res = []
    xmin_lst_all_res.append(xmin_lst_all[0])
    ymin_lst_all_res.append(ymin_lst_all[0])
    xmax_lst_all_res.append(xmax_lst_all[0])
    ymax_lst_all_res.append(ymax_lst_all[0])
    
    x_offset = 0
    y_offset = 0
    
    debugPath_new = debugPath + "_" + str(0)
    font= cv2.FONT_HERSHEY_SIMPLEX
    isOKBBox = True
    
        
    for img_id in range(1, len(img_lst)):
        img1 = img_lst[img_id-1]
        img2 = img_lst[img_id]
        
        if isDebug:
            debugPath_new = debugPath + "_" + str(img_id-1)
            
        if AlgorithmName=="calcOffsetByECC":
            x_offset,y_offset = calcOffsetByECC(img1, img2, isDebug, debugPath_new)
        elif AlgorithmName=="calcOffsetByORB":
            x_offset,y_offset = calcOffsetByORB(img1, img2, isDebug, debugPath_new)
        elif AlgorithmName=="calcOffsetByOpticalLK":
            x_offset,y_offset = calcOffsetByOpticalLK(img1,img2, xmin_lst_all[img_id-1: img_id+1],ymin_lst_all[img_id-1: img_id+1],xmax_lst_all[img_id-1: img_id+1],ymax_lst_all[img_id-1: img_id+1], labelInfo_lst, isDebug,debugPath_new) 
        else:
            x_offset = 0
            y_offset = 0
        
        if x_offset>300 or x_offset<-300 or y_offset>300 or y_offset<-300:
            y_offset=0
            x_offset=0
            
        x1_lst = []
        y1_lst = []
        x2_lst = []
        y2_lst = []

        h = img1.shape[0]
        w = img2.shape[1]
        
        for bbox_id in range( len(xmin_lst_all[img_id])):
            x1_tmp = protectXY(xmin_lst_all[img_id][bbox_id] - x_offset, w)
            y1_tmp = protectXY(ymin_lst_all[img_id][bbox_id] - y_offset, h)
            x2_tmp = protectXY(xmax_lst_all[img_id][bbox_id] - x_offset, w)
            y2_tmp = protectXY(ymax_lst_all[img_id][bbox_id] - y_offset, h)
            '''print ("orig: ", xmin_lst_all[img_id][bbox_id], ymin_lst_all[img_id][bbox_id], xmax_lst_all[img_id][bbox_id], ymax_lst_all[img_id][bbox_id], x_offset, y_offset)
            print ("dest: ", x1_tmp, y1_tmp, x2_tmp, y2_tmp)'''
            
            if x1_tmp >= x2_tmp or y1_tmp>= y2_tmp:
                isOKBBox = False
                
            x1_lst.append( x1_tmp )
            y1_lst.append( y1_tmp )
            x2_lst.append( x2_tmp )
            y2_lst.append( y2_tmp )
        
        xmin_lst_all_res.append(x1_lst)
        ymin_lst_all_res.append(y1_lst)
        xmax_lst_all_res.append(x2_lst)
        ymax_lst_all_res.append(y2_lst)
        if isDebug:
            if img_id==1:
                debugPath_new1 = debugPath + "_" + str(img_id-1)+ "_"+AlgorithmName + "_org.jpg"
                cv2.imwrite(debugPath_new1, img1)
                
                debugPath_new1 = debugPath + "_" + str(img_id)+ "_"+AlgorithmName +"_org.jpg"
                img2_tmp = copy.deepcopy(img2)
                cv2.putText(img2_tmp, "(" + str(x_offset) + ","+ str(y_offset)+")", (50,50), font, 1,(0,0,255),2)
                cv2.imwrite(debugPath_new1, img2_tmp)
            else:
                debugPath_new1 = debugPath + "_" + str(img_id)+ "_"+AlgorithmName +"_org.jpg"
                img2_tmp = copy.deepcopy(img2)
                cv2.putText(img2_tmp, "(" + str(x_offset) + ","+ str(y_offset)+")", (50,50), font, 1,(0,0,255),2)
                cv2.imwrite(debugPath_new1, img2_tmp)
    
    if isOKBBox:            
        return xmin_lst_all_res,ymin_lst_all_res,xmax_lst_all_res,ymax_lst_all_res
    else:
        return xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all