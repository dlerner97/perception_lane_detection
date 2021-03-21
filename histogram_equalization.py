import cv2
import numpy as np
from os import system
from video_handler import VideoHandler

class HistogramEqualization:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_hist(frame):
        hist = np.zeros((256), int)
        for px in frame.ravel():
            hist[px] += 1
            
        return hist
      
    @staticmethod       
    def equalize_frame(frame):
        if len(frame.shape) > 2:
            print("Please input a grayscale image")
            return
            
        h, w = frame.shape[:2]
        hist = HistogramEqualization.get_hist(frame)
        frame_arr = frame.ravel()
        len_frame_arr = len(frame_arr)
        cdf = [round(255*np.sum(hist[:ind+1])/len_frame_arr) for ind in range(256)]
        for ind in range(len_frame_arr):
            frame_arr[ind] = cdf[frame_arr[ind]]
            
        return frame
        
        
if __name__ == "__main__":
    handle = VideoHandler("Night Drive - 2689.mp4", 50)
    
    def body(frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = HistogramEqualization.equalize_frame(frame_gray)
        cv2.imshow("frame", frame)
    
    handle.run(body)
            
