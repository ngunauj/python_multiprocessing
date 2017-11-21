import multiprocessing
import time
 
def worker(interval):
    n = 5
    while n > 0:
        print("The time is {0}".format(time.ctime()))
        time.sleep(interval)
        n -= 1

def worker_1(interval):
    #print "worker_1"
    print("worker_1 begin time is {0}".format(time.ctime()))
    time.sleep(interval)
    print("worker_1 end time is {0}".format(time.ctime()))
 
def worker_2(interval):
    #print "worker_2"
    print("worker_2 begin time is {0}".format(time.ctime()))
    time.sleep(interval)
    print("worker_2 end time is {0}".format(time.ctime()))
 
def worker_3(interval):
    #print "worker_3"
    print("worker_3 begin time is {0}".format(time.ctime()))
    time.sleep(interval)
    print("worker_3 end time is {0}".format(time.ctime()))
 
if __name__ == "__main__":
    p1 = multiprocessing.Process(target = worker_1, args = (2,))
    p2 = multiprocessing.Process(target = worker_2, args = (2,))
    p3 = multiprocessing.Process(target = worker_3, args = (2,))
 
    p1.start()
    p2.start()
    p3.start()
 
    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))