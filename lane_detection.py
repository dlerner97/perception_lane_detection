import cv2
import numpy as np
from video_handler import VideoHandler
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

class LaneDetectionHandler(VideoHandler):
    def __init__(self, video_name, scale_percent, K, D, lane_output_shape=(400, 600)) -> None:
        super().__init__(video_name, scale_percent=scale_percent)
        self.K = K
        self.D = D
        self.lane_output_shape = lane_output_shape
        h,w = super().get_img_shape()
        self.ideal_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D,(w,h), 1, (w,h))
        
    def set_projective_transform(self, selected_corners):
        # h, w = self.get_img_shape()
        h, w = self.lane_output_shape
        start_x, end_x = int(.3*w), int(.5*w)
        start_y, end_y = 0, int(.5*h)
        
        selected_corners = np.float32(selected_corners)
        birds_eye_corners = np.float32([[start_x,end_y], [end_x, end_y], [end_x, start_y], [start_x, start_y]])
        self.M = cv2.getPerspectiveTransform(selected_corners, birds_eye_corners)
    
    def apply_projective_transform(self, img):
        return cv2.warpPerspective(img, self.M, self.lane_output_shape)[::-1,:]
    
    def undistort(self, img):
        return cv2.undistort(img, self.K, self.D, None, self.ideal_cam_mat)
    
    def get_frame_n(self, frame_num):
        return self.undistort(super().get_frame_n(frame_num))
    
    def get_next_frame(self):
        ret, frame = super().get_next_frame()
        try:
            cv2.imshow("frame", frame)
            frame = self.undistort(frame)   
            frame = self.apply_projective_transform(frame)
        except TypeError:
            pass
        finally:
            return ret, frame
    
    def get_lane_corners(self):
        print("Please select four corners for homography in the clockwise direction.")
        print("Use W/A/S/D to move the circle and set the four corners.")
        print("Use E/R to lower/upper the pixel movement per key press.")
        print("Select V to set the image to a new image and x to reset selection.")
        print("Lastly select Q to quit. If four corners are chosen, the program will print the selected corners.")
        
        n = 1
        img = self.get_frame_n(n)        
        h, w = img.shape[:2]
        circle_x = w//2
        circle_y = h//2
        res = 5
        corners = []
        
        try:
            while True:
                img_cp = img.copy()
                               
                if corners:
                    for coord in corners:                        
                        from_x, to_x = coord[0]-200, coord[0]+200
                        from_y, to_y = coord[1]-200, coord[1]+200
                        cv2.line(img_cp, (from_x, coord[1]), (to_x, coord[1]), (255,0,0), 1) 
                        cv2.line(img_cp, (coord[0], from_y), (coord[0], to_y), (255,0,0), 1)
                        cv2.circle(img_cp, coord, 3, (0,255,0), -1)
                
                cv2.circle(img_cp, (circle_x, circle_y), 1, (0,0,255), -1)
                cv2.imshow("corner finder", img_cp)
                key = cv2.waitKey(1) & 0xff
                
                if key == ord('w'):
                    circle_y -= res
                elif key == ord('s'):
                    circle_y += res
                elif key == ord('a'):
                    circle_x -= res
                elif key == ord('d'):
                    circle_x += res  
                elif key == ord('r'):
                    res += 1
                elif key == ord('e'):
                    res -= 1
                    if res < 1:
                        res = 1
                elif key == ord('v'):
                    n += 1
                    img = self.get_frame_n(n)
                elif key == ord('x'):
                    corners = []
                elif key == 32:
                    corners.append((circle_x, circle_y))
                    circle_x = w//2
                    circle_y = h//2
                    res = 5          
                elif key == ord('q'):
                    break
                
            if len(corners) != 4:
                print("Please input exactly 4 points")
                return
                
            print(f"The four corners in clockwise are: {corners}")
            return corners
        except KeyboardInterrupt:
            return

class LaneDetection:
    def __init__(self) -> None:
        pass
    
    def detect_lanes(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", frame_gray)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame_t = cv2.filter2D(frame_gray, -1, kernel)
        cv2.imshow("blurred", frame_t)
        
        _, frame_t = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", frame_t)
        
        white_px = np.argwhere(frame_t)
        unique, counts = np.unique(white_px[:,1], return_counts=True)
        arg_maxes = np.argsort(-counts, 0)
        desired_maxes = int(round(.5*len(arg_maxes)))
        sorted_white_px_bool = np.isin(white_px[:, 1], unique[arg_maxes[:desired_maxes]])
        sorted_white_px = white_px[sorted_white_px_bool]
        # frame[sorted_white_px[:,0], sorted_white_px[:,1]] = (0,0,255)
        
        _, w = frame_gray.shape
        right_sorted_white_px_bool = sorted_white_px[:,1] > w//2
        left_sorted_white_px_bool = np.invert(right_sorted_white_px_bool)
        
        right_sorted_white_px = sorted_white_px[right_sorted_white_px_bool]
        left_sorted_white_px = sorted_white_px[left_sorted_white_px_bool]
        
        # frame[right_sorted_white_px[:,0], right_sorted_white_px[:,1]] = (0,0,255)
        # frame[left_sorted_white_px[:,0], left_sorted_white_px[:,1]] = (0,255,0)
        
        

        return frame


if __name__ == "__main__":
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                  [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    D = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
    
    vid_name = 'data_1/data_vid1.avi'
    
    # This function takes all of the images and patches them into a video. It then saves the resulting video. Therefore,
    # this function must only be run if the video has not been generated yet 
    # VideoHandler.gen_video_frm_imgs(vid_name, 'data_1/data', 10, 302, '.png', 10)
    
    lane_detector = LaneDetection()
    handle = LaneDetectionHandler(vid_name, 50, K, D)
    
    # These two lines bring up a prompt that allows the user to pick coordinates for the corners
    # handle.get_lane_corners()
    
    selected_corners = [(348, 173), (450, 173), (520, 247), (190, 247)]
    handle.set_projective_transform(selected_corners)
    
    def body(frame):
        frame = lane_detector.detect_lanes(frame)
        cv2.imshow("img", frame)
        
    handle.run(body)
    
    
    
    
    
    
    
    
    