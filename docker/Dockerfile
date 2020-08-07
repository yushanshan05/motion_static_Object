# Use Caffe2 image as parent image
FROM motion_static_predict:1.0.0

# install git
RUN apt-get update
RUN apt-get install git -y

RUN git clone https://github.com/yushanshan05/motion_static_Object  /motion_static_Object


COPY motion_static.pth /motion_static_Object/model/

# start restful service
WORKDIR /motion_static_Object
RUN export PYTHONPATH=`pwd`/test
CMD python app.py
EXPOSE 8081
