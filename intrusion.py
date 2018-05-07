import pandas as pd
import numpy as np
import cv2,imutils,os,sys
from imutils.object_detection import non_max_suppression
import configparser,datetime
import json
import subprocess

class intrusion(object):
	
	def __init__(self, video_path, city_name, location_name = " ", ROI=None):
		
		self.__video_channel = video_path
		self.__video_name = os.path.splitext(os.path.basename(video_path))[0]
		
		self.__ROI = ROI 
		# clean city name given by user    
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


		# injecting schema here
		# sensor meta info
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

		# event info
		Event = {
		'Alert_id' : 1,
		'TypeDescription' : 'Somebody enter the virtual fencing'
		}

		event_json_file_name = 'event.json'
		with open(os.path.join(json_file_path, event_json_file_name), 'w') as f:
			json.dump(Event, f)


		# 
		self.__path_to_output = os.path.join(predef_dir_structure_path, 'output')
		self.__path_to_log = os.path.join(predef_dir_structure_path, 'log')
		

		# create output directory if doesn't exists already
		if not os.path.exists(self.__path_to_output):
			os.makedirs(self.__path_to_output)
		


		print("Done Init!")


				  
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
		
		current_date = str(datetime.datetime.now().date())

		# visual_output_path = self.__path_to_output+"/visual_files/"+str(datetime.datetime.now().date())+"/"
		visual_output_path = os.path.join(self.__path_to_output, current_date)
		
		# create output directory if doesn't exists already
		if not os.path.exists(visual_output_path):
			os.makedirs(visual_output_path)

		log_file = open(os.path.join(self.__path_to_log, current_date+".txt"), "a+")
		
		log_file.write(self.start_time+"-->Starting Intrusion detection at time\n")
		
		cap = cv2.VideoCapture(self.__video_channel)
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
		log_file.close()
		
		

		while True:

			
			current_date = str(datetime.datetime.now().date())

			# visual_output_path = self.__path_to_output+"/visual_files/"+str(datetime.datetime.now().date())+"/"
			visual_output_path = os.path.join(self.__path_to_output, current_date)

			# create output directory if doesn't exists already
			if not os.path.exists(visual_output_path):
				os.makedirs(visual_output_path)

			log_file = open(os.path.join(self.__path_to_log, current_date+".txt"), "a+")

			# log_file = open(os.path.join(self.__path_to_log, current_date+".txt"), "a+")

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
					# print((xA, yA), (xB, yB))
					cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
				#------------------------------if intrusion happens------------------------
				#print(intrusion_started_time,countdown_time)
				if any(res):
					if ((intrusion_started_time is None) and (countdown_time==0)):
						#print('creating video instance')
						intrusion_started_time = datetime.datetime.now()
						# video_file_name = visual_output_path+intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+".avi"
						video_file_name = os.path.join(visual_output_path, intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+'.avi')
						# print video_file_name
						# out = cv2.VideoWriter(visual_output_path+intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+".avi",fourcc, 20.0, (image.shape[1],image.shape[0]))
						out = cv2.VideoWriter(video_file_name, fourcc, 15.0, (image.shape[1],image.shape[0]))
					countdown_time = 45  # extra time fow which video is going to be written

					## Do all shit here...event happened...
					## more than one human detected
					
					num_humans = sum(res)
					out.write(image)
					cv2.putText(image,"number of humans= "+str(num_humans),(40,40), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)
					# writing log file
					data = {
						'ReportedTime' : str(datetime.datetime.now()),
						'VideoURL(FilePath)' : str(video_file_name)
					}
					# print("PICK:" + str(pick))
					x1 = pick[0][0]
					y1 = pick[0][1]
					x2 = pick[0][2]
					y2 = pick[0][3]

					detected_roi = []
					human_list = []
					for idx, (xA, yA, xB, yB) in enumerate(pick):
						# print "idx: " + str(idx)
						# print((xA, yA), (xB, yB))
						
						human_list.append({
								"name" : "human_"+str(idx),
								"Point1": { "X": str(x1), "Y": str(y1) }, 
								"Point2": { "X": str(x1+x2), "Y": str(y1) }, 
								"Point3": { "X": str(x1), "Y": str(y1+y2) },  
								"Point4": { "X": str(x2), "Y": str(y2) }

							})

						detected_roi.append(
							{
							"humans" : 	
							{
							"Point1": { "X": str(x1), "Y": str(y1) }, 
							"Point2": { "X": str(x1+x2), "Y": str(y1) }, 
							"Point3": { "X": str(x1), "Y": str(y1+y2) },  
							"Point4": { "X": str(x2), "Y": str(y2) }

								}
							}
						)

					# print "detected_roi: " + str(detected_roi)

					cdata = { "input": { "data": { "SensorMetaInfo": {"CameraID": "test_video_1", "Feature_id": 1, "Product_category_id": 1, "ServerId": "vijaywada_PC_01", "Lx": "10.233N", "Ly": "70.1212S", "CameraDescription": "cisco_cam_type_1", "LongDescription": "low range camera"}, "Event": { "EventID": 1, "EventDescription": "Somebody enter the virtual fencing" }, "ROI_drawn": { "Point1": { "X": str(bbox1[0]), "Y": str(bbox1[1]) }, "Point2": { "X": str(bbox1[0]+bbox1[2]), "Y": str(bbox1[1]) }, "Point3": { "X": str(bbox1[0]), "Y": str(bbox1[1]+bbox1[3]) }, "Point4": { "X": str(bbox1[0]+bbox1[2]), "Y": str(bbox1[1]+bbox1[3]) } }, "Data": { "number_of_humans": str(num_humans), 

					"detected_roi": detected_roi,

						"CapturedTime": str(datetime.datetime.now()), "VideoURL": "http://<ip-address>"+str(video_file_name) } } }, "configName": "IntrusionDetection", "groupName": "VideoAnalytics" }

					# subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94", "-H", "x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR", "-d", json.dumps(cdata)])



					# with open("ads.json" 'w') as f:
						# json.dumps(cdata, f)

					# log_file.write(str(datetime.datetime.now())+"-->"+str(num_humans)+" humans detected within ROI\n")
					log_file.write('Number of humans within ROI: ' + str(num_humans) +', detected at time: ' + str(datetime.datetime.now()) + ', '+video_file_name  + ', ROI: ' + str(pick) +'\n')
				else :
					
					intrustion_stopped = True
					intrusion_started_time = None
					if out is not None:
						countdown_time-=1
						if (countdown_time==0 ):
							# print(json.dumps(cdata))
							# subprocess.call(['curl -X POST http://52.74.189.153:9090/api/v1/source/getInputData', '-H', 'Cache-Control: no-cache', '-H', 'Content-Type: application/json',  ], shell=True)
							# subprocess.call(['curl -X POST   http://52.74.189.153:9090/api/v1/source/getInputData', '-H', 'Cache-Control: no-cache', '-H', 'Content-Type: application/json', '-H', 'Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94', '-H', 'x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR', '-d', '{ "input": { "data": { "SensorMetaInfo": {"CameraID": "test_video_1", "Feature_id": 1, "Product_category_id": 1, "ServerId": "vijaywada_PC_01", "Lx": "10.233N", "Ly": "70.1212S", "CameraDescription": "cisco_cam_type_1", "LongDescription": "low range camera"}, "Event": { "EventID": 1, "EventDescription": "Somebody enter the virtual fencing" }, "ROI_drawn": { "Point1": { "X": "10", "Y": "132" }, "Point2": { "X": "26", "Y": "132" }, "Point3": { "X": "26", "Y": "148" }, "Point4": { "X": "10", "Y": "148" } }, "Data": { "number_of_humans": 2, "detected_roi": { "Point1": { "X": "112", "Y": "312" }, "Point2": { "X": "34", "Y": "356" } }, "CapturedTime": "2018-02-26T10:23:51", "VideoURL": "http://<ip-address>/<path-to-output>/bangalore/indranagar/society-2/intrusion/output visual_files/2018_04_23_18_01/2018_04_23_18_01_11.avi" } } }, "configName": "IntrusionDetection", "groupName": "VideoAnalytics" }'], shell=True)
							# subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94", "-H", "x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR", "-d", '{ "input": { "data": { "SensorMetaInfo": {"CameraID": "test_video_1", "Feature_id": 1, "Product_category_id": 1, "ServerId": "vijaywada_PC_01", "Lx": "10.233N", "Ly": "70.1212S", "CameraDescription": "cisco_cam_type_1", "LongDescription": "low range camera"}, "Event": { "EventID": 1, "EventDescription": "Somebody enter the virtual fencing" }, "ROI_drawn": { "Point1": { "X": "10", "Y": "132" }, "Point2": { "X": "26", "Y": "132" }, "Point3": { "X": "26", "Y": "148" }, "Point4": { "X": "10", "Y": "148" } }, "Data": { "number_of_humans": 2, "detected_roi": { "Point1": { "X": "112", "Y": "312" }, "Point2": { "X": "34", "Y": "356" } }, "CapturedTime": "2018-02-26T10:23:51", "VideoURL": "http://<ip-address>/<path-to-output>/bangalore/indranagar/society-2/intrusion/output visual_files/2018_04_23_18_01/2018_04_23_18_01_11.avi" } } }, "configName": "IntrusionDetection", "groupName": "VideoAnalytics" }'])
							
							subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94", "-H", "x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR", "-d", json.dumps(cdata)])
							print cdata
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
			
			# print("ENDING HERE")
			log_file.close()

			detected_roi = {}

# intr = intrusion('test_video_1.mp4', 'kanpur')
# intr.detect_intrusion(plot=True)