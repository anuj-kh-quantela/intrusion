import pandas as pd
import numpy as np
import cv2,imutils,os,sys
from imutils.object_detection import non_max_suppression
import configparser,datetime
import json

class intrusion(object):
    
    # def __init__(self,video_channel,city_name='vijaywada',ROI=None,log_path=None,config_path=None,output_path=None):
    def __init__(self, video_path, city_name, location_name = " ", ROI=None):
    	
        self.__video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        self.video_channel = video_path
        self.__ROI = ROI 
            
        self.__city_name = city_name.strip().lower().replace(" ", "_")
        # clean location name given by user
        self.__location_name = location_name.strip().lower().replace(" ", "_")
        # get absolute path to project directory
        self.__dir_hierarchy = os.getcwd()
        # extract project path and project name from absolute path
        self.__project_path, self.__project_name = os.path.split(self.__dir_hierarchy)

        # list of all the predefined directories to be made 
        predef_dir_structure_list = ['input', 'output', 'log', 'config']
        # define root path for the above directory structure
        predef_dir_structure_path = os.path.join(self.__project_path, self.__city_name, self.__location_name, self.__project_name)

        # make all the directories on the defined root path
        for folder in predef_dir_structure_list:        
            dir_structure = os.path.join(predef_dir_structure_path, folder)

            if not os.path.exists(dir_structure):
                os.makedirs(dir_structure)


        # THIS IS NEW
        json_file_path = predef_dir_structure_path

        SensorMetaInfo = {
            'CameraID' : self.__video_name,
            'ServerId' : 'vijaywada_PC_01',
            'Product_category_id' : 1,
            'Feature_id' : 1,
            'Lx' : '10.233N',
            'Ly' :  '70.1212S'

        }

        sensor_meta_info_json_file_name = 'SensorMetaInfo.json'
        with open(os.path.join(json_file_path, sensor_meta_info_json_file_name), 'w') as f:
            json.dump(SensorMetaInfo, f)

        Event = {
        'Alert_id' : 1,
        'TypeDescription' : 'Somebody enter the virtual fencing'
        }

        event_json_file_name = 'event.json'
        with open(os.path.join(json_file_path, event_json_file_name), 'w') as f:
            json.dump(Event, f)



        self.__user_output_path = os.path.join(predef_dir_structure_path, 'output')
        self.__path_to_output = self.__user_output_path
        self.__log_path = os.path.join(predef_dir_structure_path, 'log')


        # create output directory if doesn't exists already
        if not os.path.exists(self.__path_to_output):
            os.makedirs(self.__path_to_output)
        


        print("Done Init!")


        
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

    def detect_intrusion(self, plot=False):
        
        """
        plot: To show intermediate images
        """
        self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        # self.makedirs_1(self.output_path+"visual_files/"+self.start_time+"/",exist_ok=True)
        visual_output_path = self.__path_to_output+"/visual_files/"+self.start_time+"/"
        
        # create output directory if doesn't exists already
        if not os.path.exists(visual_output_path):
            os.makedirs(visual_output_path)

        log_file =open(self.__log_path+ "/" +self.start_time+".txt", "a+")
        
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
        bbox1 = self.__ROI
        out = None
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        countdown_time = 0
        roi_created_flag = False
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
                    if not any(bbox1):
                        print("NO ROI SELECTED, Exiting!")
                        sys.exit(0)
                    print("Selected ROI Coordinates: " + str(bbox1))
                    
                    self.__ROI = bbox1
                    roi_created_flag = True
                    # self.write_config()

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
                        video_file_name = visual_output_path+intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+".avi"
                        out = cv2.VideoWriter(visual_output_path+intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+".avi",fourcc, 20.0, (image.shape[1],image.shape[0]))
                    countdown_time = 45  # extra time fow which video is going to be written

                    ## Do all shit here...event happened...
                    ## more than one human detected
                    
                    num_humans = sum(res)
                    out.write(image)
                    cv2.putText(image,"number of humans= "+str(num_humans),(40,40), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)
                    # writing log file
                    data = {


                        'ReportedTime' : datetime.datetime.now(),
                        'CapturedTime' : intrusion_started_time
                        # 'VideoURL(FilePath)' : visual_output_path+intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+".avi" 
                    }

                    # log_file.write(str(datetime.datetime.now())+"-->"+str(num_humans)+" humans detected within ROI\n")
                    log_file.write('Number of humans within ROI: ' + str(num_humans) +', detected at time: ' + str(datetime.datetime.now()) + ', '+video_file_name  + ', ROI: ' + str(pick) +'\n')
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
            

            
