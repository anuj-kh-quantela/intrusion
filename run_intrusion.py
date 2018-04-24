import multiprocessing
from intrusion import intrusion
import numpy as np



def wrapper(video_channel, city_name, location_name, roi):
		
	intr = intrusion(video_channel, city_name, location_name, ROI=roi)
	# intr = intrusion('test_video_1.mp4', 'vijaywada', 'datacenter', ROI=roi)
	# intr = intrusion('test_video_1.mp4', 'bangalore', "indranagar/society-2")
	intr.detect_intrusion(plot=True)

video_channel_arr = ['rtsp://192.168.20.9/6d801f1f-a9aa-449a-85d3-88608e5ee67b/6d801f1f-a9aa-449a-85d3-88608e5ee67b_vs1?token=6d801f1f-a9aa-449a-85d3-88608e5ee67b^LVERAMOTD^50^26^26^1657790795^d660cf85eebea453b0c933b63025aedeb9c22fea&username=admin']
location_name_arr = ['datacenter']
roi_arr = [(228, 57, 149, 154)]

if __name__ == '__main__':
	
	jobs = []
	for i in range(len(location_name_arr)):
		video_channel = video_channel_arr[i]
		location_name = location_name_arr[i]
		roi = roi_arr[i]

		p = multiprocessing.Process(target=wrapper, args=(video_channel, 'vijaywada', location_name, roi,))
		jobs.append(p)
		p.start()



