import caffe
import numpy as np
import yaml
import cv2
import random

#import pickle
#import time

class PythonDataInputLayer(caffe.Layer):
    def setup(self, bottom, top):
      layer_params = yaml.load(self.param_str)
      self.batchnum = int(layer_params['batchnum'])
      livefilelist = layer_params['livefilelist']
      imposterfilelist = layer_params['imposterfilelist']
      self.imagelivelist=open(livefilelist).readlines()
	  random.shuffle(self.imagelivelist)
	  self.imageimposterlist=open(imposterfilelist).readlines()
	  random.shuffle(self.imageimposterlist)
      self.liveidx = 0
      self.imposteridx = 0
	  self.imagesize = 112
      top[0].reshape(self.batchnum, 6, self.imagesize, self.imagesize)
      top[1].reshape(self.batchnum)
        
   		
    def forward(self, bottom, top):
      tmp_batch=np.zeros([self.batchnum, 6, self.imagesize, self.imagesize], dtype=np.float32)
	  live_num = self.batchnum/2
	  imposter_num = self.batchnum-live_num
	  while(live_num>0):
	    image_flip=False
	    if(random.uniform(0,1)>0.5):
		  image_flip=True
        m_items=self.imagelivelist[self.liveidx].rstrip().split('\t')
        self.liveidx+=1
        self.liveidx%=len(self.imagelivelist)
        img=cv2.imread(m_items[0],1)
        if(img is None):
          continue
        face_x=int(m_items[1])
	    face_y=int(m_items[2])
        face_w=int(m_items[3])
	    face_h=int(m_items[4])
	    face_conf=float(m_items[5])
	    face_cx=face_x+face_w/2
	    face_cy=face_y+face_h/2
	    face_w=int(face_w*random.uniform(0.8,1.2))
	    face_h=face_w
	    face_x=face_cx-face_w/2
	    face_y=face_cy-face_y/2
	    if(face_x<0):
	      continue
	    if(face_y<0):
		  continue
	    if(face_conf<0.45):
		  continue
	    if(face_x+face_w>img.shape[1]):
		  continue
	    if(face_y+face_h>img.shape[0]):
		  continue
	    aug_resize=random.randint(56,224)
	    face_img=img[face_y:face_y+face_h,face_x:face_x+face_w].copy()
	    face_img=cv2.resize(face_img,(aug_resize,aug_resize))
        face_img=cv2.resize(face_img, (self.imagesize, self.imagesize))
	    if(image_flip):
		  face_img=cv2.flip(face_img,1)
	    tmp_batch[self.batchnum/2-live_num,0:3,:,:]=face_img.transpose(2,0,1)
	    img=cv2.copyMakeBorder(img, face_h, face_h, face_w, face_w, cv2.BORDER_REFLECT)
	    img=img[face_y:face_y+3*face_h, face_x:face_x+3*face_w]
	    img=cv2.resize(img,(aug_resize*3,aug_resize*3))
        img=cv2.resize(img, (self.imagesize, self.imagesize))
	    if(image_flip):
		  img=cv2.flip(img,1)
	    tmp_batch[self.batchnum/2-live_num,3:6,:,:]=img.transpose(2,0,1)
	    live_num-=1
	  while(imposter_num>0):
	    image_flip=False
	    if(random.uniform(0,1)>0.5):
		  image_flip=True
	    m_items=self.imageimposterlist[self.imposteridx].rstrip().split('\t')
	    self.imposteridx+=1
	    self.imposteridx%=len(self.imageimposterlist)
	    img=cv2.imread(m_items[0],1)
	    if(img is None):
		  continue
        face_x=int(m_items[1])
	    face_y=int(m_items[2])
        face_w=int(m_items[3])
	    face_h=int(m_items[4])
	    face_conf=float(m_items[5])
	    face_cx=face_x+face_w/2
	    face_cy=face_y+face_h/2
	    face_w=int(face_w*random.uniform(0.8,1.2))
	    face_h=face_w
	    face_x=face_cx-face_w/2
	    face_y=face_cy-face_y/2
	    if(face_x<0):
		  continue
	    if(face_y<0):
		  continue
	    if(face_conf<0.45):
		  continue
	    if(face_x+face_w>img.shape[1]):
		  continue
	    if(face_y+face_h>img.shape[0]):
		  continue
	    aug_resize=random.randint(90,448)
	    face_img=img[face_y:face_y+face_h,face_x:face_x+face_w].copy()
	    face_img=cv2.resize(face_img,(aug_resize,aug_resize))
        face_img=cv2.resize(face_img, (self.imagesize, self.imagesize))
	    if(image_flip):
		  face_img=cv2.flip(face_img,1)
	    tmp_batch[self.batchnum-imposter_num,0:3,:,:]=face_img.transpose(2,0,1)
	    img=cv2.copyMakeBorder(img, face_h, face_h, face_w, face_w, cv2.BORDER_REFLECT)
	    img=img[face_y:face_y+3*face_h, face_x:face_x+3*face_w]
	    img=cv2.resize(img,(aug_resize*3,aug_resize*3))
        img=cv2.resize(img, (self.imagesize, self.imagesize))
	    if(image_flip):
		  img=cv2.flip(img,1)
	    tmp_batch[self.batchnum-imposter_num,3:6,:,:]=img.transpose(2,0,1)
	    imposter_num-=1
	  top[0].reshape(self.batchnum, 6, self.imagesize, self.imagesize)
	  top[0].data[...]=tmp_batch-128
	  tmp_labels=np.zeros(self.batchnum, dtype=np.float32)
	  live_num = self.batchnum/2
	  tmp_labels[0:live_num]=0
	  tmp_labels[live_num:self.batchnum]=1
      top[1].reshape(self.batchnum)
	  top[1].data[...]=tmp_labels 
            
    def backward(self, top, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
