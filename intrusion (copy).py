import pandas as pd
import numpy as np
import cv2,imutils,os,sys
from imutils.object_detection import non_max_suppression
import configparser,datetime


class intrusion (object):
    
    def __init__(self,video_channel,city_name='vijaywada',ROI=None,log_path=None,config_path=None,output_path=None):
        self.__project_name = "intrusion"
        self.__city_name = city_name
        self.video_channel = video_channel
        self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.ROI =ROI 
        
        if log_path is None:
            self.makedirs_1("../"+self.__city_name+"/"+self.__project_name+"/log_files/",exist_ok=True)
            self.log_path = "../"+self.__city_name+"/"+self.__project_name+"/log_files/"

            
        if output_path is None:
            self.makedirs_1("../"+self.__city_name+"/"+self.__project_name+"/output_path/",exist_ok=True)
            self.makedirs_1("../"+self.__city_name+"/"+self.__project_name+"/output_path/flat_files/",exist_ok=True)
            self.makedirs_1("../"+self.__city_name+"/"+self.__project_name+"/output_path/visual_files/",exist_ok=True)
            self.output_path = "../"+self.__city_name+"/"+self.__project_name+"/output_path/"
            
        if config_path is None:
            self.makedirs_1("../"+self.__city_name+"/"+self.__project_name+"/config_path/",exist_ok=True)
            self.config_path = "../"+self.__city_name+"/"+self.__project_name+"/config_path/"
            self.write_config()

                
            
    def write_config(self):
        config = configparser.ConfigParser()
        config.optionxform = str
        config[self.__project_name +"_"+self.__city_name] = {}
        config[self.__project_name +"_"+self.__city_name]['video_channel'] = self.video_channel
        config[self.__project_name +"_"+self.__city_name]['log_path'] = self.log_path
        config[self.__project_name +"_"+self.__city_name]['config_path'] = self.config_path
        config[self.__project_name +"_"+self.__city_name]['output_path'] = self.output_path
        config[self.__project_name +"_"+self.__city_name]['ROI'] = str(self.ROI)
        #writing config file
        with open(self.config_path+self.__project_name+".ini", 'w') as configfile:
            config.write(configfile)
        
    def makedirs_1(self,path,exist_ok=True):
        try :
            os.makedirs(path)
        except Exception as e:
            if not exist_ok:
                raise Exception(e)
                
        
                
    def check_intersection(self,a,b):
        """
        a : rectangle in (x,y,w,h)
        """
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w<0 or h<0: 
            return False,() # or (0,0,0,0) ?
        return True,(x, y, w, h)
    def detect_intrusion(self,plot=False):
        """
        plot: To show intermediate images
        """
        self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.makedirs_1(self.output_path+"visual_files/"+self.start_time+"/",exist_ok=True)
        visual_output_path = self.output_path+"visual_files/"+self.start_time+"/"
        
        log_file =open(self.log_path+self.start_time+".txt", "a+")
        
        log_file.write(self.start_time+"-->Starting Intrusion detection at time\n")
        cap = cv2.VideoCapture(self.video_channel)
        ret = True
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        if plot:
            cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        log_file.write(str(datetime.datetime.now())+"-->Detector Initiated\n")
        
        intrusion_started_time = None
        bbox1 = self.ROI
        out = None
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        countdown_time = 0
        while True:
            current_time = datetime.datetime.now()
            try:
                ret,frame = cap.read()
                if not (cap.isOpened() and ret):
                    break


                image = imutils.resize(frame, width=min(400, min(frame.shape[:2])))
                orig = image.copy()
                if bbox1 is None:
                    bbox1 = cv2.selectROI('select_ROI', image)
                    cv2.destroyWindow('select_ROI')
                    self.ROI = bbox1
                    self.write_config()
                log_file.write(str(datetime.datetime.now())+"--> ROI created "+str(bbox1)+"\n")


                # detect people in the image
                (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
                #(rects, weights) = hog.detect(image, winStride=(4, 4),padding=(8, 8))
        #         rects = [i.tolist()+[100,100] for i in rects]



            #     # apply non-maxima suppression to the bounding boxes using a
            #     # fairly large overlap threshold to try to maintain overlapping
            #     # boxes that are still people
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                cv2.rectangle(image,(bbox1[0],bbox1[1]),(bbox1[0]+bbox1[2],bbox1[1]+bbox1[3]),(255,0,0),1)

                    #rects = [[i[0],i[1],i[2]-i[1],i[3]-i[2]] for i in pick]
                    #rects = [cv2.rectangle() for i in pick]
                res = []
                for rect in pick :
                    res.append(self.check_intersection(np.array(rect),np.array(bbox1))[0])
                #print(res)
                for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                #------------------------------if intrusion happens------------------------
                #print(intrusion_started_time,countdown_time)
                if any(res):
                    if ((intrusion_started_time is None) and (countdown_time==0)):
                        #print('creating video instance')
                        intrusion_started_time = datetime.datetime.now()
                        out = cv2.VideoWriter(visual_output_path+intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+".avi",fourcc, 20.0, (image.shape[1],image.shape[0]))
                    countdown_time = 45  # extra time fow which video is going to be written

                    ## Do all shit here...event happened...
                    ## more than one human detected
                    
                    num_humans = sum(res)
                    out.write(image)
                    cv2.putText(image,"number of humans= "+str(num_humans),(40,40), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)
                    #writing log file
                    log_file.write(str(intrusion_started_time)+"-->"+str(num_humans)+" humans detected within ROI\n")
                else :
                    intrustion_stopped = True
                    intrusion_started_time = None
                    if out is not None:
                        countdown_time-=1
                        if (countdown_time==0 ):
                            out.release()
                            out = None

                # draw the final bounding boxes
                if plot:
                    cv2.imshow("detection",image)
                    k = cv2.waitKey(1)
                    if k==27:
                        cv2.destroyAllWindows()
                        cap.release()
                        break
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e,exc_type, fname, exc_tb.tb_lineno)
                log_file.write(str(datetime.datetime.now())+"--> Exception"+str(e)+str(exc_type)+" "+str(fname)+" "+ str(exc_tb.tb_lineno))
            

            
        
        
            
        
intr = intrusion('rtsp://192.168.20.9/6d801f1f-a9aa-449a-85d3-88608e5ee67b/6d801f1f-a9aa-449a-85d3-88608e5ee67b_vs1?token=6d801f1f-a9aa-449a-85d3-88608e5ee67b^LVERAMOTD^50^26^26^1657790795^d660cf85eebea453b0c933b63025aedeb9c22fea&username=admin',ROI=(228, 57, 149, 154))
intr.detect_intrusion()