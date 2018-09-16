# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 20:27:08 2018

@author: ahmed
"""

import cv2
import numpy as np
import Camera
import Lanes
import traceback
from Perspective import Perspective

outpath = "test_images"
camera = Camera.Camera()
camera.calibrate('camera_cal', 'calibration*.jpg', 6, 9, visualize=False)
#camera.turnOn()
camera.setOpMode(2)
camera.setVideoCapture('project_video.mp4') #'challenge_video.mp4') #
persp = Perspective()
img, src, dst = persp.getTransformParametersLong()
camera.setPerspectiveTransform(img, src, dst)
prev_left_fit = None
prev_right_fit = None
laneSmoother = Lanes.LaneSmoother()

# Controls whether to just extract static images from video for development and testing,
# or to actually process the video through the pipeline.
mode = None #'extract'

cv2.namedWindow('Road Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Road Video', 600,400)
cv2.namedWindow('Lanes', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lanes', 600,400)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(camera.cap.get(3))
frame_height = int(camera.cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#cv2.VideoWriter_fourcc('*divx')
#out = cv2.VideoWriter('C:/Yaser/Udacity/CarND-Term1/project_video_processed.mp4', -1, 10, (frame_width,frame_height))

out = cv2.VideoWriter("C:/Yaser/Udacity/CarND-Term1/project_video_out.avi", -1, 10, (frame_width,frame_height), isColor=True)
frame_count = 0
try:
    while(True):
        frame_count += 1
        ret, roadFrame = camera.getNextFrame()
        if(ret == True):
            if(mode=='extract'): # Extract static images from video for development and testing
                cv2.imshow('Road Video', roadFrame.frame)
                cv2.imwrite(outpath + "/project_test{:02d}".format(frame_count) + ".jpg", roadFrame.frame)
                
            else: # Process video
                lane = roadFrame.getLane(prev_left_fit, prev_right_fit, visualize=False) # encapsulates logic of undistort, threshold, perspective transform, identify lines, polyfit etc.
                lane = laneSmoother.smooth(lane)
                
                # Display the resulting frame
                highlightedLane = roadFrame.highlightLane(lane.left_fitx, lane.right_fitx, lane.ploty)
                
                prev_left_fit = lane.left_coeff
                prev_right_fit = lane.right_coeff
                rofc = lane.getRadiusOfCurvature()
                print("rofc = ", rofc)
                lane_center_pix, lane_center_xm = lane.getCenter()
                vehicleOffset_xm = roadFrame.getVehicleOffset()
                
                # Display frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                text1 = "Radius of Curv: " + str(np.round(np.mean(rofc),2))
                text2 = "Vehicle offset from Center: " + str(np.round(vehicleOffset_xm,2))
                cv2.putText(highlightedLane, text1, (40, 40), font, 1, (0, 150, 0), 2)
                cv2.putText(highlightedLane, text2, (40, 70), font, 1, (0, 150, 0), 2)
                
                # Write the frame into the file 'project_video_processed.mp4'
                out.write(highlightedLane)
                
                cv2.imshow('Lanes',lane.lane_img)
                cv2.imshow('Road Video', highlightedLane)
            
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
except: # Exception as e:
    print(traceback.format_exc())
    cv2.waitKey(0)

camera.cap.release()
out.release()
cv2.destroyAllWindows()