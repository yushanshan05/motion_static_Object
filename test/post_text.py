#coding=utf-8
import requests
import time
import json
import base64
import cv2
import os
from scipy import misc
#from test_case import info_lst_case9

#info_lst=info_lst_case9

num = 1
mtcnn_elapsed = 0
facenet_elapsed = 0
emotion_elapsed = 0
eye_elapsed = 0
angle_elapsed = 0
alltime = 0

i = 0
start = time.time()

s = requests

root_path = "test/img"
txt_path = "post_text.txt"
img_name = 'img'

class_list=['digger','truckbig','trucksmall','tanker','forklift']

with open(txt_path, "r") as f:
    txt_data = f.readlines()#按行读取，一行为一组图片
    for i in range(len(txt_data)):#处理一组
        info_lst=[] 
        one_img_group = txt_data[i].split(',')
        group_name = one_img_group[0].split()[0]#当前图片组名
        group_path = os.path.join(img_name,group_name)
        if os.path.exists(group_path):
            if len(one_img_group) <= 2:
                continue
            #group_bbox_list = getOneGroupBbox(one_img_group)#一组图片中所有bbox对的列表
            #group_img_path_list = getGroupImgName(group_path,group_name)
            #一组图片中所有图片路径，读取用
            #一组图片中所有图片输出路径，处理后输出用
            cls_list = []
            xmin_list = []
            ymin_list = []
            xmax_list = []
            ymax_list = []
            score_list = []
            for i in range(len(one_img_group)-1):
                if len(one_img_group[i+1].split())<2:
                    one_img_dic = {"imgPath": os.path.join(root_path,one_img_group[i+1].split()[0]),"cls":[],"score":[],"xmin":[],"ymin":[],"xmax":[],"ymax":[]}
                    info_lst.append(one_img_dic)
                else:
                    for j in range(int((len(one_img_group[i+1].split())-1)/6)):
                        if one_img_group[i+1].split()[5+j*6]  in class_list:
                            xmin_list.append(int(one_img_group[i+1].split()[1+j*6]))
                            ymin_list.append(int(one_img_group[i+1].split()[2+j*6]))
                            xmax_list.append(int(one_img_group[i+1].split()[3+j*6]))
                            ymax_list.append(int(one_img_group[i+1].split()[4+j*6]))
                            cls_list.append(one_img_group[i+1].split()[5+j*6])
                            score_list.append(one_img_group[i+1].split()[6+j*6])
                    one_img_dic = {"imgPath": os.path.join(root_path,one_img_group[i+1].split()[0]),"cls":cls_list,"score":score_list,"xmin":xmin_list,"ymin":ymin_list,"xmax":xmax_list,"ymax":ymax_list}
                    info_lst.append(one_img_dic)
                cls_list = []
                score_list = []
                xmin_list = []
                ymin_list = []
                xmax_list = []
                ymax_list = []
        data={"data":info_lst,"param":{"iouth":0.2}}
        my_json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        start1 = time.time() 
        r = s.post('http://0.0.0.0:8081/user', headers=headers,data = my_json_data,)
        end1 = time.time() - start1
        print (end1)
        #print type(r)
        #print (r)
        #print type(r.json())            

        print (r.json())
        #add plot
        #img = cv2.imread(os.path.join(imagepath,file))
        data= {}
        data = r.json()
        datalist = []
        datalist = data['data']
        #print(len(datalist))
        for j in range(len(datalist)):
            singledata = {}
            #boxdict = {}
            singledata = datalist[j]
            #print(singledata)
            '''boxdict = singledata['bbox']
            xmin = boxdict['xmin']
            ymin = boxdict['ymin']
            xmax = boxdict['xmax']
            ymax = boxdict['ymax']
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,255,0))'''
            
            '''font= cv2.FONT_HERSHEY_SIMPLEX
            strname = singledata['cls']
            strscore = singledata['score']
            #print (type(strscore))
            print (strscore)'''
            #cv2.putText(img, strname + str(strscore), (5,5), font, 1,(0,0,255),2)
        #print(os.path.join(imagepath_out,file))
        #cv2.imwrite(os.path.join(imagepath_out,file), img)
        end = time.time() - start
        #print (end)

        #plot
        #imagepath = '/data/ligang/detectron/Detectron-master/restful/vis/806_180507070134.jpg'
        #img = cv2.imread(imagepath)
        #cv2.rectangle(img, (136,63), (765,474),3)
        #cv2.rectangle(img, (130,50), (537,239),3)
        #cv2.imwrite('./001_new.jpg', img)       
'''
################################################################
############################# curl #############################
curl -X POST 'http://192.168.200.213:9527/user' -d '{"data":"/opt/ligang/detectron/Detectron-master/restful/vis/180523_0006_6000.jpg"}' -H 'Content-Type: application/json'


curl -X POST 'http://192.168.200.213:9527/user' -d '{"data":"https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1526895699811&di=5ce6acbcfe8f1d93fe65d3ae8eb3287d&imgtype=0&src=http%3A%2F%2Fimg1.fblife.com%2Fattachments1%2Fday_130616%2F20130616_e4c0b7ad123ca263d1fcCnkYLFk97ynn.jpg.thumb.jpg"}' -H 'Content-Type: application/json'
'''