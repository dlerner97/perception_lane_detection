import cv2
from os import system

from typing_extensions import final

# This class handles simple image and video tasks
class VideoBuffer():
    def __init__(self, framerate = 100, scale_percent=100):
        self.frames = []
        self.framerate = framerate
        self.scale_percent = scale_percent

    # Write frames to feed and save video
    def save(self, video_name, isColor = False):
        shape = self.frames[0].shape
        size = (shape[1], shape[0])
        
        if video_name[-4:] != '.avi':
            video_name = video_name + '.avi'
        
        videowriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), self.framerate, size, isColor)
        for f in self.frames:
            videowriter.write(f)
        videowriter.release()

class VideoHandler:
    def __init__(self, video_name, video_out_name, scale_percent=100, framerate=10) -> None:
        system('cls')
        self.video_feed = cv2.VideoCapture(video_name)
        self.scale_percent = scale_percent
        
        _, frame = self.video_feed.read()
        self.video_feed.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = self.resize_frame(frame, self.scale_percent)
        self.shape = frame.shape[:2]
        self.video_out_name = video_out_name
        self.framerate = framerate
        
    @staticmethod
    def gen_video_frm_imgs(vid_name, img_rel_folder, name_char_len, last_num, img_type = '.png', framerate=10):
        buffer = VideoBuffer(framerate)
        
        if img_rel_folder[-1] != '/':
            img_rel_folder = img_rel_folder + '/'
        
        for i in range(last_num+1):
            str_i = str(i)
            img_name = img_rel_folder + '0'*(name_char_len-len(str_i)) + str_i + img_type
            img = cv2.imread(img_name)
            buffer.frames.append(img)
            
        buffer.save(vid_name, True)
        
    # Resizes each length image at a given percent. E.g. 200 will double each dimension and 50 will half it
    @staticmethod
    def resize_frame(frame, scale_percent=100):
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    
    def get_frame_n(self, frame_num):        
        curr_frame_num = self.video_feed.get(cv2.CAP_PROP_POS_FRAMES)
        self.video_feed.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = self.video_feed.read()
        self.video_feed.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_num)
        frame = self.resize_frame(frame, self.scale_percent)
        return frame
    
    def get_img_shape(self):
        return self.shape
    
    def get_num_frames(self):
        return self.video_feed.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def get_next_frame(self):
        ret, frame = self.video_feed.read()

        if not ret:
            return None, None
        
        frame = self.resize_frame(frame, self.scale_percent)
        return ret, frame
    
    def run(self, body):
        buffer = VideoBuffer(self.framerate, 50)
        wait = 0
        try:
            while True:
                ret, frame = self.get_next_frame()

                if not ret:
                    break
            
                frame_processed = body(frame)
                buffer.frames.append(frame_processed)
                
                k = cv2.waitKey(wait) & 0xff
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    if wait == 0:
                        wait = 100
                    else:
                        wait = 0
        except KeyboardInterrupt:
            pass
        finally:
            buffer.save(self.video_out_name, True)
            
    def __del__(self):
        self.video_feed.release()