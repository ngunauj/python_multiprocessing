import caffe
import numpy
import cv2
import random
import pickle
import time
from collections import deque
import sys


class MouthDataLayer():
    def setup(self):
        self.imgname = 0
        self.minibatch_size = 128
        self.seq_len = 10
        self.rows = 48
        self.cols = 48
 
    def forward(self):
        #start=time.clock()
        
        tmp_top0=numpy.zeros((self.minibatch_size, self.seq_len, self.rows, self.cols ),numpy.float32)
        tmp_top1=numpy.zeros((self.minibatch_size, 1),numpy.int32)
        
        mini_idx=0
        while True:
            if mini_idx == self.minibatch_size:
                break
            if random.randint(1,100)<10:
            #### move ####
                if random.randint(1,10)>5:
                    idx=random.randint(0,self.len_is-1)
                else:
                    idx=random.randint(self.len_is-1, self.is_move_num-1)
                tmp_video=self.is_move_video[idx]
                if random.randint(1,10)>8:
                    ifok,tmp_seq=self.get_std_seq(tmp_video)
                else:
                    ifok,tmp_seq=self.get_jump_seq(tmp_video)
                if ifok==False:
                    continue
                
             	tmp_seq=self.augment(tmp_seq)
                
                #self.display(tmp_seq, True)
		
		tmp_top0[mini_idx,:,:,:]=tmp_seq
                tmp_top1[mini_idx,:]=0
                mini_idx+=1
            #### stay ####
            else:
                if random.randint(1,10)>5:
                    idx=random.randint(0,self.len_not-1)
                else:
                    idx=random.randint(self.len_not-1, self.not_move_num-1)
                tmp_video=self.not_move_video[idx]
                if random.randint(1,10)>8:
                    ifok,tmp_seq=self.get_std_seq(tmp_video)
                else:
                    ifok,tmp_seq=self.get_stay_seq(tmp_video)
                if ifok==False:
                    continue
                
		tmp_seq=self.augment(tmp_seq)

                #self.display(tmp_seq, False)
                
		tmp_top0[mini_idx,:,:,:]=tmp_seq
                tmp_top1[mini_idx,:]=1
                mini_idx+=1
        #end=time.clock()
        #print 'fetch time:' , (end - start)
        return tmp_top0, tmp_top1

    def backward(self):
        """This layer does not propagate gradients."""
        pass

    def reshape(self):
        """Reshaping happens during the call to forward."""
        pass

def train_net(argv):

	caffe.set_mode_gpu()
	caffe.set_device(int(argv[1]))
	solver = caffe.SGDSolver(argv[2])
	
	#data_poll=deque()
	#label_poll=deque()
	data_poll=list()
	label_poll=list()
	
	datalayer = MouthDataLayer()
	datalayer.setup()
	
	iter_num = solver.iter
	while True:
	
	    data,label=datalayer.forward()
	    
	    if len(data_poll)>500:
	        for x in range(len(data)/2):
	            idx1 = random.randint(0,len(data)-1)
	            idx2 = random.randint(0,len(data_poll)-1)
	            #data[x,:,:,:]=data_poll.popleft()
	            #label[x,:]=label_poll.popleft()
	            data[idx1,:,:,:]=data_poll[idx2]
	            label[idx1,:]=label_poll[idx2]
	
	
	    solver.net.blobs['data'].data[...] = data
	    solver.net.blobs['label'].data[...] = label
	    
	    solver.step(1)
	
	    iter_num = solver.iter
	    if iter_num%10000==0 and iter_num>0:
	        solver.snapshot()
	        print 'save checkpoint'
	    if iter_num%100==0:
	        print '............................................................data_poll size:', len(label_poll)
	        print '............................................................minibatch wrong:', wrong_label.shape[0]
	        print '............................................................minibatch wrong label mean:', numpy.mean(wrong_label)
	    if iter_num>1000000:
	        break
	        print 'done'
	
if __name__ == '__main__':
    train_net(sys.argv)
    print sys.argv
