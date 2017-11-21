import multiprocessing
import time

def writer_proc(q):
    global num      
    while q.qsize() < 10:
        q.put(num, block = False)
        num += 1

def reader_proc(q):      
    try:
        #time.sleep(2)         
        print 'get' , q.get(block = False)
        #print q.qsize() 
    except:         
        pass
 
if __name__ == "__main__":
    q = multiprocessing.Queue()
    num = 100
    
    while True:
        writer = multiprocessing.Process(target=writer_proc, args=(q,))  
        writer.start()   
 
        reader = multiprocessing.Process(target=reader_proc, args=(q,))  
        reader.start()  
        time.sleep(1)
        
    reader.join()  
    writer.join()