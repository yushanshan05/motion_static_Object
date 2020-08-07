#coding=utf-8
import os
import cv2
import numpy as np
import time
from cv2.xfeatures2d import matchGMS
from scipy.cluster.vq import kmeans,vq

def getOffsetFromKmean(point_lst, orient_lst_tmp_tmp, fw, isDebug=False):
    point_num = len(point_lst)   
    num = int(2*len(point_lst)/3)           #### 聚类个数
    ##kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True)
    
    max_num = 2
    dw = 0
    dh = 0
    distortion = -1
    # 使用kmeans进行聚类分析，设置终止条件为执行10次迭代或者精确度epsilon=1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)   ############ modify
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    point_arr = np.array(point_lst)
    #point_arr = point_arr.reshape((point_arr, 2))
    # 运用kmeans
    # 返回值有紧密度、标志和聚类中心。标志的多少与测试数据的多少是相同的
    # cv2.kmeans(data，K， bestLabels，criteria，attempt，flags) attempts：使用不同的起始标记来执行算法的次数。算法会返回紧密度最好的标记。紧密度也会作为输出被返回
    
    min_v = 1000
    isOK = False
    kmean_num = 0
    for i in range(2,num):
        compactness, labels, centers = cv2.kmeans(point_arr, i, None, criteria, 10, flags)
        if compactness<min_v:
            min_v = compactness
            labels_tmp = labels
            centers_tmp = centers
            isOK = True
            kmean_num = i

    dw = 0
    dh = 0
    if isOK:
        if isDebug:
            for i in range(kmean_num):
                #print ("kmean: ", i)
                fw.write("kmean: ", i)
                fw.write("\n")
                for id in range(len(labels_tmp)):
                    if labels_tmp[id]==i:
                        #print (id, point_lst[id])
                        fw.write(id, point_lst[id])
                        fw.write("\n")
        
        max_value = -1
        max_id = -1  
        kmean_index_lst = []
        for i in range(kmean_num):
            lst = []
            for id in range(len(labels_tmp)):
                if labels_tmp[id]==i:
                    lst.append(id)
                    
            if len(lst)>max_value:
                max_value = len(lst)
                max_id = i
            kmean_index_lst.append(lst)

        kmean_index_max_lst = []
        for id in range(len(kmean_index_lst)):   
            if max_value==len(kmean_index_lst[id]):
                kmean_index_max_lst.append(id)
                
        max_num = 0
        if len(kmean_index_max_lst)==1:
            for id in range(len(labels_tmp)):
                if max_id==labels_tmp[id]:
                    if isDebug:
                        #print (point_lst[id])
                        fw.write("max: ", point_lst[id])
                        fw.write("\n")
                    dw = dw + point_lst[id][0]
                    dh = dh + point_lst[id][1]
                    max_num = max_num + 1
            dw = int(dw / max_value)
            dh = int(dh / max_value)
        else:
            dw_tmp = 0
            dh_tmp = 0
            min_dis_id = 0
            min_dis = 655535
            for i in range(len(kmean_index_max_lst)):
                id = kmean_index_max_lst[i]
                center_x = centers_tmp[id][0]
                center_y = centers_tmp[id][1]

                dis = 0
                for j in range(len(centers_tmp)):
                    x = centers_tmp[j][0]
                    y = centers_tmp[j][1]
                    dis = dis + np.sqrt((x-center_x)*(x-center_x) +(y-center_y)*(y-center_y))
                if min_dis>dis:
                    min_dis = dis
                    min_dis_id = id

            for i in range(len(kmean_index_lst[min_dis_id])):
                id = kmean_index_lst[min_dis_id][i]
                dw = dw + point_lst[id][0]
                dh = dh + point_lst[id][1]

            dw = int(dw/ len(kmean_index_lst[min_dis_id]))
            dh = int(dh/ len(kmean_index_lst[min_dis_id]))
    else:
        return 0,0
        
    return dw,dh,True

def getXYOffset(good_new,good_old,src2,match_point_num,fw,isDebug=False,debugPath='./'):
    if isDebug:
        color = np.random.randint(0,255,(100,3))
    
    orient_lst = []
    dw_lst = []
    dh_lst = []
    img_tmp = copy.deepcopy(src2)
    mask = np.zeros_like(src2)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        x2,y2 = new.ravel() #ravel()函数用于降维并且不产生副本
        x1,y1 = old.ravel()
        if isDebug:
            mask = cv2.line(mask, (x1,y1),(x2,y2), color[i].tolist(), 2)
            img_tmp = cv2.circle(img_tmp,(x2,y2),5,color[i].tolist(),-1)
    
        dw = x2-x1
        dh = y2-y1
        o = round((dh)/(dw+0.00000001),2)
        orient_lst.append(o)
        dw_lst.append(dw)
        dh_lst.append(dh)
        
    if isDebug:
        img_tmp1 = cv2.add(img_tmp,mask)
        save_path = debugPath + "_pt_calcOffsetByOpticalLK.jpg"
        cv2.imwrite(save_path, img_tmp1)
        
    if len(orient_lst)<match_point_num:
        print ("point num <{}", match_point_num)
        return 0,0
    
    point_lst = []
    orient_lst_tmp_tmp = []
    orient_lst_tmp = copy.deepcopy( orient_lst )
    orient_lst_tmp_id = np.argsort( orient_lst_tmp )
    for id in range(len(orient_lst_tmp_id)):
        id_tmp = orient_lst_tmp_id[id]
        point_lst.append( [dw_lst[id_tmp], dh_lst[id_tmp]] )
        if isDebug:
            orient_lst_tmp_tmp.append(orient_lst_tmp[id_tmp]) 
            #print ("sort h/w  dw  dh: ", orient_lst_tmp[id_tmp], dw_lst[id_tmp], dh_lst[id_tmp])
 
    dw, dh, dis = getOffsetFromKmean(point_lst, orient_lst_tmp_tmp, fw, isDebug)

    return dw, dh, True   
    
def getMaskFromBBox(src2, xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all, labelInfo_lst): #按照2张写的代码
    mask = np.ones_like(src2)
    for img_id in range(len(xmin_lst_all)):
        for bbox_id in range(len(xmin_lst_all[img_id])):
            x1 = xmin_lst_all[img_id][bbox_id]
            y1 = ymin_lst_all[img_id][bbox_id]
            x2 = xmax_lst_all[img_id][bbox_id]
            y2 = ymax_lst_all[img_id][bbox_id]
            h1 = y2 - y1
            w1 = x2 - x1

            img1 = np.zeros((h1,w1,3),dtype=int)
            mask[y1:y2,x1:x2] = img1[0:h1,0:w1]
            
    #######处理图片左上角打印的时间、右下角摄像头的区域
    for id in range(len(labelInfo_lst)):
        bbox = labelInfo_lst[id]
        if len(bbox)<4:
            print ("bbox error: ", bbox)
            continue
        h1 = bbox[3]-bbox[1]
        w1 = bbox[2]-bbox[0]
        img1 = np.zeros((h1,w1,3),dtype=int)
        mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = img1[0:h1,0:w1]
        
    return mask

'''
xmin_lst_all = [[1159,194,98], [1172,180,144,320]]
ymin_lst_all = [[956,644,1008],[957,649,1010,640]]
xmax_lst_all = [[1374,557,283],[1385,423,302,594]]
ymax_lst_all = [[1047,783,1078],[1048,773,1078,767]]'''
def calcOffsetByOpticalLK(src1,src2, xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all, labelInfo_lst=[[60,55,900,140], [1340,955,1920,1030]], isDebug=False,debugPath='./'):
    match_point_num=4
    
    gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    
    mask = getMaskFromBBox(src2, xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all, labelInfo_lst)
    if isDebug:
        save_path = debugPath + "_mask_calcOffsetByOpticalLK.jpg"
        cv2.imwrite(save_path , mask*src2)
        fw = open("calcOffsetByOpticalLK.txt", "a+")
        
    feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
    lk_params = dict(winSize  = (15,15),maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv2.goodFeaturesToTrack(gray1, mask=mask[:,:,0], **feature_params)
    if p0 is None:
        return 0,0
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)#计算新的一副图像中相应的特征点额位置
    good_new = p1[st==1]
    good_old = p0[st==1]
    x_offset, y_offset = getXYOffset(good_new, good_old, src2, match_point_num, fw, isDebug, debugPath)    
    return x_offset, y_offset

def warp_pos(pos, warp_matrix):
    p1 = np.array([pos[0, 0], pos[0, 1], 1]).reshape(3, 1)
    p2 = np.array([pos[0, 2], pos[0, 3], 1]).reshape(3, 1)
    p1_n = np.dot(warp_matrix, p1).reshape(1, 2)
    p2_n = np.dot(warp_matrix, p2).reshape(1, 2)
    return np.concatenate((p1_n, p2_n), 1).reshape(1, -1) 

def get_warp_pyramid(im1, im2, nol, criteria_, warp_mode):
    
    # pyr_start_time = timeit.default_timer()
    init_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    warp = init_warp
    warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32) ** (1 - nol)

    # construct grayscale pyramid
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    gray1_pyr = [gray1]
    gray2_pyr = [gray2]

    for level in range(nol):
        #print(gray1_pyr[0].shape)
        #print("fx:", 1 / 2.)
        gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx=1 / 2., fy=1 / 2.,
                                       interpolation=cv2.INTER_AREA))
        gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx=1 / 2., fy=1 / 2.,
                                       interpolation=cv2.INTER_AREA))

    # run pyramid ECC
    for level in range(nol):        
        if level != nol - 1:
            # if True:
            try:
                cc, warp = cv2.findTransformECC(gray1_pyr[level], gray2_pyr[level],
                                            warp, warp_mode, criteria_, inputMask=None, gaussFiltSize=1)
            except Exception as e:
                # error
                str_err = 'ERR cv2.findTransformECC'
                print(str_err)
                return warp, False
           
        # if level != nol-1:  # scale up for the next pyramid level
        warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)    

    return warp, True  #torch.from_numpy(warp)

    

def calcOffsetByECC(src1, src2,isDebug=False,debugPath='./'):  
    #start_time1=time.time()
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.00001) 
    pyramid_nol = 5  
    warp_mode =  cv2.MOTION_EUCLIDEAN   
    warp_matrix, isOK = get_warp_pyramid(src1, src2, pyramid_nol, criteria, warp_mode) 
    if not isOK:
        #print("ECC time:",time.time()-start_time1)  
        return 0,0
        
    pt= np.array([[100,100,100,100]])
    pos = warp_pos(pt, warp_matrix)                    
    x1_new = int(pos[0][0].item())
    y1_new = int(pos[0][1].item())
    x2_new = int(pos[0][2].item())
    y2_new = int(pos[0][3].item())
    
    x_offset=x1_new-100
    y_offset=y1_new-100     
   
    #print("ECC time:",time.time()-start_time1)  
    
    return x_offset,y_offset
    
def calcOffsetByORB(src1, src2,isDebug=False,debugPath='./'):#如果报错，安装pip install opencv-contrib-python
    orb = cv2.ORB_create(10000)#生成特征向量
    orb.setFastThreshold(0)
    kp1, des1 = orb.detectAndCompute(src1, None)
    kp2, des2 = orb.detectAndCompute(src2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)
    matches_gms = matchGMS(src1.shape[:2], src2.shape[:2], kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)
    x_sum = 0
    y_sum = 0
    #print (len(matches_gms))
    for i in range(len(matches_gms)):
       left = kp1[matches_gms[i].queryIdx].pt
       right = tuple(sum(x) for x in zip(kp2[matches_gms[i].trainIdx].pt, (src1.shape[1], 0)))
       x_offset_ = tuple(map(int, left))[0]-tuple(map(int, right))[0]+1920
       y_offset_ = tuple(map(int, left))[1]-tuple(map(int, right))[1]
       x_sum = x_sum + x_offset_
       y_sum = y_sum + y_offset_
    x_offset = int(x_sum/len(matches_gms))
    y_offset = int(y_sum/len(matches_gms))
    if isDebug == True:
        height = max(src1.shape[0], src2.shape[0])
        width = src1.shape[1] + src2.shape[1]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:src1.shape[0], 0:src1.shape[1]] = src1
        output[0:src2.shape[0], src1.shape[1]:] = src2[:]
        for i in range(len(matches_gms)):
            left = kp1[matches_gms[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches_gms[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))
        debug_out_path = os.path.join(debugPath+'_calcOffsetByORB.jpg')
        cv2.imwrite(debug_out_path,output)
        #print (x_offset,y_offset)
    return x_offset,y_offset

if __name__ == '__main__':   
    im1_path="/data/AI/yushan/motion_static/images/8001/JS-002_33_20200708154639.jpg"
    im2_path="/data/AI/yushan/motion_static/images/8001/JS-002_33_20200708154650.jpg"
    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)
    
    #calcOffsetByECC(img1, img2, True, debugPath='/data/AI/yushan/motion_static/debug_info/8001_1')
    #calcOffsetByORB(img1, img2, True, debugPath='/data/AI/yushan/motion_static/debug_info/8001_1')

    xmin_lst_all = [[1159,194,98], [1172,180,144,320]]
    ymin_lst_all = [[956,644,1008],[957,649,1010,640]]
    xmax_lst_all = [[1374,557,283],[1385,423,302,594]]
    ymax_lst_all = [[1047,783,1078],[1048,773,1078,767]]
    calcOffsetByOpticalLK(img1,img2, xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all, labelInfo_lst=[[60,55,900,140], [1340,955,1920,1030]], isDebug=False,debugPath='./')        
            
            