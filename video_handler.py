import cv2
from os import system

class VideoHandler:
    def __init__(self, video_name, scale_percent=100) -> None:
        system('cls')
        self.video_feed = cv2.VideoCapture(video_name)
        self.scale_percent = scale_percent
      
    @staticmethod
    def resize_frame(grid, scale_percent=100):
        width = int(grid.shape[1] * scale_percent / 100)
        height = int(grid.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(grid, dim, interpolation = cv2.INTER_AREA)    
    
    def run(self, body):
        wait = 0
        try:
            while True:
                ret, frame = self.video_feed.read()

                if not ret:
                    break
                
                frame = self.resize_frame(frame, self.scale_percent)
                body(frame)
                
                k = cv2.waitKey(wait) & 0xff
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    if wait == 0:
                        wait = 1
                    else:
                        wait = 0
        except KeyboardInterrupt:
            pass
            
    def __del__(self):
        self.video_feed.release()