#<=============================== Imports ===================================>#
import cv2
import numpy as np
from os import system
from video_handler import VideoHandler

#<=============================== HistogramEqualization Class Definition ===================================>#
class HistogramEqualization:
    
    """
    Histogram Equalization Class
    
    Args: gamma 
    
    Self-made class for histogram equalization with gamma correction.
    
    """
    
    def __init__(self, gamma=10) -> None:
        self.hist = np.zeros((256), int)
        self.gamma = gamma
    
    # Generates a histogram for a given frame (only works with grayscaled img)
    def get_hist(self, frame):
        self.hist = np.zeros((256), int)
        for px in frame.ravel():
            self.hist[px] += 1
      
    # Equalizes frame with a given gamma correction 
    def equalize_frame(self, frame):
        if len(frame.shape) > 2:
            print("Please input a grayscale image")
            return
            
        self.get_hist(frame)
        frame_arr = frame.ravel()
        len_frame_arr = len(frame_arr)
        
        # Define CDF
        cdf = [round(255*(np.sum(self.hist[:ind+1])/len_frame_arr)**self.gamma) for ind in range(256)]
        
        # Apply CDF
        for ind in range(len_frame_arr):
            frame_arr[ind] = cdf[frame_arr[ind]]
            
        return frame
        
#<=============================== Main ===================================>#
if __name__ == "__main__":
    
    handle = VideoHandler("Night Drive - 2689.mp4", "Night Drive - 2689_processed", 50)
    hist_equal = HistogramEqualization()
    
    def body(frame):
        cv2.imshow("orig frame", frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = hist_equal.equalize_frame(frame_gray)
        cv2.imshow("frame", frame)
        return frame
    
    handle.run(body)
            
