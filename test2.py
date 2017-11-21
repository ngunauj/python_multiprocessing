import multiprocessing
import time
 
class ClockProcess(multiprocessing.Process):
    def __init__(self, interval):
        multiprocessing.Process.__init__(self)
        self.interval = interval
    def run(self):
        #print "worker_1"
        print("worker_1 begin time is {0}".format(time.ctime()))
        time.sleep(self.interval)
        print("worker_1 end time is {0}".format(time.ctime()))

if __name__ == '__main__':
    p = ClockProcess(2)
    p.daemon = True
    p.start()
    p.join()
    print 'first'