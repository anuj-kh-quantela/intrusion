from intrusion import intrusion

roi = (228, 57, 149, 154)
intr = intrusion('rtsp://192.168.20.9/6d801f1f-a9aa-449a-85d3-88608e5ee67b/6d801f1f-a9aa-449a-85d3-88608e5ee67b_vs1?token=6d801f1f-a9aa-449a-85d3-88608e5ee67b^LVERAMOTD^50^26^26^1657790795^d660cf85eebea453b0c933b63025aedeb9c22fea&username=admin', 'vijaywada', 'datacenter' ,ROI=roi)
# intr = intrusion('test_video_1.mp4', 'vijaywada', 'datacenter' ,ROI=roi)
# intr = intrusion('test_video_1.mp4', 'bangalore', "indranagar/society-2")
intr.detect_intrusion(plot=False)