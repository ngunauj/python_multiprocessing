import glob
import os
import cv2
import concurrent.futures
'''
read image and do nothing
'''

def run(path):
  img = cv2.imread(path)
  if img is None:
    print 'read error'
  else:
    print 'ok'

with concurrent.futures.ProcessPoolExecutor() as executor:
  pathlist = glob.glob("/home/guan/Desktop/out_0.7/*.jpg")
  executor.map(run, pathlist)

  
