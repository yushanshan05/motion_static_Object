#coding=utf-8
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from model import Model
from flask import request,Response
from scipy import misc
import json
import urllib
import cv2
import os
import time
#import core.infer_simple_test as infer_test
#from infer_simple_test import Model
from model import Model
app = Flask(__name__)


@app.route('/user', methods=['POST'])
def info():
    # logger add
    
    formatter = logging.Formatter("[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/motionstatic.log', maxBytes=20000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    ip = request.remote_addr
    info_str = 'IP:' + ip
    logger.info(info_str)
    info_str = 'model_path:' + weights_path
    logger.info(info_str)
    #imagepath = request.form.getlist('data')
    #imagepath = request.form.get("data",type=str,default=None)
    start_time = time.time()
    js = request.get_json()
    #js is not dict return []
    js=eval(str(js))
    #return json init
    out_json = {"data":[]}
    imgPath_all = []
    img_all = []
    img_all_bgr = []
    cls_lst_all = []
    score_lst_all = []
    xmin_lst_all = []
    ymin_lst_all = []
    xmax_lst_all = []
    ymax_lst_all = []
    
    iou_th_input = 0.2   # 默认值为0.2     
    #score_th_input = 1.0  
    
    info_str = ''
    if isinstance(js,dict) and 'data' in js:
    
        all_lst_temp = js.get('data',None)

        for j in range(len(all_lst_temp)):
            all_lst1 = all_lst_temp[j]
            keys_lst = list(all_lst1.keys())
            print(keys_lst)
            for index_key in range( len(keys_lst) ):
                if keys_lst[index_key]=="imgPath":
                    imgPath_all.append( all_lst1["imgPath"] )
                if keys_lst[index_key]=="cls":
                    cls_lst_all.append( all_lst1["cls"] )
                if keys_lst[index_key]=="score":
                    score_lst_all.append( all_lst1["score"] )
                if keys_lst[index_key]=="xmin":
                    xmin_lst_all.append( all_lst1["xmin"] )
                if keys_lst[index_key]=="ymin":
                    ymin_lst_all.append( all_lst1["ymin"] )
                if keys_lst[index_key]=="xmax":
                    xmax_lst_all.append( all_lst1["xmax"] )
                if keys_lst[index_key]=="ymax":
                    ymax_lst_all.append( all_lst1["ymax"] )
          
        all_lst_temp1 = js.get('param',None)
        keys_lst1 = list(all_lst_temp1.keys())
        for index_key in range(len(keys_lst1)):            
            if keys_lst1[index_key]=="iouth":
                iou_th_input = all_lst_temp1["iouth"] 
        
        
        info_str = 'images path:' + ','.join(imgPath_all)
        logger.info(info_str)
        info_str = 'cls:{}'.format(cls_lst_all)
        logger.info(info_str) 
        info_str = 'xmin:{}'.format(xmin_lst_all)
        logger.info(info_str) 
        info_str = 'ymin:{}'.format(ymin_lst_all)
        logger.info(info_str) 
        info_str = 'xmax:{}'.format(xmax_lst_all)
        logger.info(info_str)
        info_str = 'ymax:{}'.format(ymax_lst_all)
        logger.info(info_str)
    else:
        logger.warning('post has not data or data-key!!!')
        end_time = time.time() - start_time
        logger.info('predict time:{}'.format(end_time))
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
        
    for i in range(len(imgPath_all)):   
        imagepath = imgPath_all[i]
        if (imagepath.startswith("https://") or imagepath.startswith("http://") or imagepath.startswith("file://")):
            imagefile = urllib.urlopen(imagepath)
            status=imagefile.code
            # url
            if(status==200): 
                image_data = imagefile.read()
                image_name = os.path.basename(imagepath)
                #new_imagepath = filepath+"/"+image_name
                new_imagepath = image_name
                with open(new_imagepath, 'wb') as code:
                    code.write(image_data)
                #img_np = misc.imread(new_imagepath)
                img_np = cv2.imread(new_imagepath)  #read image by cv2 ,the same as /tool/test_net.py
                if img_np is None:
                    logger.warning('the images is NONE!!!')
                    end_time = time.time() - start_time
                    logger.info('predict time:{}'.format(end_time))
                    logger.removeHandler(handler)
                    handler.close()
                    return json.dumps(out_json)
            else:
                logger.warning('the image is not download on internet!!!')
                end_time = time.time() - start_time
                logger.info('predict time:{}'.format(end_time))
                logger.removeHandler(handler)
                handler.close()
                return json.dumps(out_json)
        # path 
        else:
            if not os.path.exists(imagepath):
                logger.warning('the image is not exists!!!')
                end_time = time.time() - start_time
                logger.info('predict time:{}'.format(end_time))
                logger.removeHandler(handler)
                handler.close()
                return json.dumps(out_json)
            else:
                #img_np = misc.imread(imagepath)
                start_readimage_time = time.time()
                img_np_bgr = cv2.imread(imagepath)  #read image by cv2 ,the same as /tool/test_net.py
                end_readimage_time = time.time() - start_readimage_time
                logger.info('readimage time:{}'.format(end_readimage_time))
                if img_np_bgr is None:
                    logger.warning('the images is NONE!!!')
                    end_time = time.time() - start_time
                    logger.info('predict time:{}'.format(end_time))
                    
                    logger.removeHandler(handler)
                    handler.close()
                    return json.dumps(out_json)
                else:                    
                    img_np = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
        
        img_all.append(img_np)
        img_all_bgr.append(img_np_bgr)
    
    #增加最后一张和第一张的判断            
    
    if len(img_all)<2:
        logger.info('only one image!')
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
    
    #遍历查找有效的目标个数，如果所有的有效目标个数都不超过2个，那么直接返回空
    cls_valid_dict={} 
    for cls_index in cls_lst_all:
        for cls_item in cls_index:
            if cls_item in cls_valid_dict.keys():
                cls_valid_dict[cls_item] = cls_valid_dict[cls_item] + 1
            else:
                cls_valid_dict[cls_item] = 1
    
    
    valid_flag = False    
    for item in cls_valid_dict:
        if cls_valid_dict[item] == 1:
            valid_flag = False
        else:
            valid_flag = True            
            break
    
    if  valid_flag == False:
        logger.info('all objects appear only once!')
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)   
    
    if len(img_all) >2:    
        first_img=img_all[0]
        img_all.append(first_img)
        first_img_bgr=img_all_bgr[0]
        img_all_bgr.append(first_img_bgr)
        first_cls=cls_lst_all[0]
        cls_lst_all.append(first_cls)
        first_xmin=xmin_lst_all[0]
        xmin_lst_all.append(first_xmin)
        first_ymin=ymin_lst_all[0]
        ymin_lst_all.append(first_ymin)
        first_xmax=xmax_lst_all[0]
        xmax_lst_all.append(first_xmax)
        first_ymax=ymax_lst_all[0]
        ymax_lst_all.append(first_ymax)
        first_score=score_lst_all[0]
        score_lst_all.append(first_score)
    
                
    if mm.judgeImgSizeIsOK(img_all) is False:
        logger.info('Image size is error!')
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
        
    
        
    #判断输入的信息是否匹配        
    if not (len(img_all)==len(img_all_bgr)==len(cls_lst_all)==len(score_lst_all)==len(xmin_lst_all)==len(ymin_lst_all)==len(xmax_lst_all)==len(ymax_lst_all)):
        lst1 = [str(len(img_all)),str(len(img_all_bgr)),str(len(cls_lst_all)),str(len(score_lst_all)),str(len(xmin_lst_all)),str(len(ymin_lst_all)),str(len(xmax_lst_all)),str(len(ymax_lst_all))]
        
        logger.info('len(img),len(cls),len(score),len(xmin),len(ymin),len(xmax),len(ymax):{}'.format(",".join(lst1)))
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
        
    for i in range(len(cls_lst_all)):  
        if not (len(cls_lst_all[i])==len(score_lst_all[i])==len(xmin_lst_all[i])==len(ymin_lst_all[i])==len(xmax_lst_all[i])==len(ymax_lst_all[i])):
            lst1 = [str(len(cls_lst_all[i])),str(len(score_lst_all[i])),str(len(xmin_lst_all[i])),str(len(ymin_lst_all[i])),str(len(xmax_lst_all[i])),str(len(ymax_lst_all[i]))]
            s = ','.join(lst1)
            logger.info('imageIndex,len(cls),len(score),len(xmin),len(ymin),len(xmax),len(ymax):{}'.format(','.join(lst1)))
            logger.removeHandler(handler)
            handler.close()
            return json.dumps(out_json)
    
    start_model_time = time.time()
    
    #目前暂时规定IOU最大值不超过0.2    
    iou_th_input=min(iou_th_input, 0.2)
    
    group_bbox_list_rst = mm.judgeObjectWork(img_all,img_all_bgr,cls_lst_all,score_lst_all,xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all,iou_th_input) 
    end_model_time = time.time() - start_model_time
    logger.info('model time:{}'.format(end_model_time))  

    workObjInfo = []
    
    if len(group_bbox_list_rst) > 0:
        #group_bbox_list_rst:xmin0,ymin0,xmax0,ymax0, cls,score     
        for img_index in range(len(imgPath_all)):
            cls_lst=[]
            score_lst=[]
            xmin_lst=[]
            ymin_lst=[]
            xmax_lst=[]        
            ymax_lst=[]
            
            for oneGroup in group_bbox_list_rst:
                if len(oneGroup[img_index])==6:
                    score_lst.append(oneGroup[img_index][5])
                    cls_lst.append(oneGroup[img_index][4])
                    xmin_lst.append(oneGroup[img_index][0])
                    ymin_lst.append(oneGroup[img_index][1])
                    xmax_lst.append(oneGroup[img_index][2])
                    ymax_lst.append(oneGroup[img_index][3])
            
            if len(cls_lst)>0:
                singleImgInfo_lst = {"imgPath":imgPath_all[img_index],"cls":cls_lst,"score":score_lst,"xmin":xmin_lst,"ymin":ymin_lst,"xmax":xmax_lst,"ymax":ymax_lst}
                workObjInfo.append(singleImgInfo_lst)    
        
    
    if len(workObjInfo) > 0:
        logger.info('the images predict completed!!!')
        res_log = []
        res_log.append(info_str)
        for i in range(len(workObjInfo)):
            single_data = {}
            single_data = workObjInfo[i]
            res_log.append(single_data['cls'])
        logger.info(res_log)
        out_json["data"] = workObjInfo
    else:
        logger.warning('the images has no work objects!!!')
        out_json["data"] = workObjInfo
    end_time = time.time() - start_time
    logger.info('predict time:{}'.format(end_time))
    logger.removeHandler(handler)
    handler.close()
    return json.dumps(out_json)
    
    
    
if __name__ == '__main__':

    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    logger = logging.getLogger('motionstatic')    #set root level , default is WRAINING
    logger.setLevel(logging.DEBUG)
    
    '''
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/oilsteal.log', maxBytes=10000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)     #ok  start root log
    #app.logger.addHandler(handler)  #ok  start private log
    '''
    
    
    weights_path = '/opt/motion_static/motion_static.pth'
    if not os.path.exists(weights_path):        
        weights_path = './model/motion_static.pth'     


    iou_th=0.2   # 默认值为0.2           
    mm = Model(weights_path, iou_th)
    
    #pre_predict
    '''x_position=[]
    y_position=[]
    for i in range(5):
        img_np = cv2.imread('./001_new.jpg')
        if img_np is None:
            continue
        predict_datalist = mm.predict(img_np,x_position,y_position, './001_new.jpg')'''

    app.run(host="0.0.0.0",port=8081,debug=False)   #threaded=True

    
