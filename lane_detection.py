import cv2
import time
import numpy as np
from os import system
from sklearn.cluster import KMeans
from video_handler import VideoHandler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

class LaneDetectionHandler(VideoHandler):
    def __init__(self, video_name, scale_percent, K, D, lane_output_shape=None) -> None:
        super().__init__(video_name, scale_percent=scale_percent)
        self.K = K
        self.D = D
        h,w = super().get_img_shape()
        
        if lane_output_shape == None:
            # self.lane_output_shape = (w,h)
            self.lane_output_shape = (int(.5*w),int(2.5*h/2))
            # (400, 600)
        else:
            self.lane_output_shape = lane_output_shape
            
        self.ideal_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D,(w,h), 1, (w,h))
        
    def set_projective_transform(self, selected_corners):
        h, w = self.lane_output_shape
        start_x, end_x = int(.51*w), int(.65*w)
        start_y, end_y = 0, int(1.5*h)
        # start_x, end_x = int(.5*w), int(.85*w)
        # start_y, end_y = 0, int(1.5*h)        
        
        # start_x, end_x = 0, w
        # start_y, end_y = 0, h
        
        # start_x, end_x = int(1.4*w), int(2*w)
        # start_y, end_y = 0, int(.5*h)
        
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
        frame_cp = None
        try:
            cv2.imshow("frame", frame)
                        
            frame = self.undistort(frame)
            frame_cp = frame.copy()   
            frame = self.apply_projective_transform(frame)
        except TypeError:
            pass
        finally:
            return ret, (frame, frame_cp, self.M)
    
    def get_lane_corners(self):
        print("Please select four corners for homography in the clockwise direction.")
        print("Use W/A/S/D to move the circle and set the four corners.")
        print("Use E/R to lower/upper the pixel movement per key press.")
        print("Select V to set the image to a new image and x to reset selection.")
        print("Lastly select Q to quit. If four corners are chosen, the program will print the selected corners.")
        
        n = 0
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
        self.approx_perp_dist = None
    
    def detect_lanes(self, frame):
        M = frame[-1]
        orig_frame = frame[1]
        frame = frame[0]
        
        h, w = frame.shape[:2]
        concrete = np.zeros((h, w), np.uint8)
        
        stds = np.std(frame, axis=2)
        means = np.mean(frame, axis=2)
        std_bool_indeces = stds <= 7
        mean_low_bool_indeces = means > 50
        mean_high_bool_indeces = means < 220
        bool_indeces = std_bool_indeces & mean_low_bool_indeces & mean_high_bool_indeces
        concrete[bool_indeces] = 255
        cv2.imshow("c", concrete)
              
        closing = cv2.morphologyEx(concrete, cv2.MORPH_CLOSE, np.ones((71,31)))      
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((h,31)))
        
        cv2.imshow("open", opening)
        cv2.imshow("close", closing)
        # cv2.imshow("dilate", dilate)
        
        frame_masked = cv2.bitwise_and(frame, frame, mask=opening)
        cv2.imshow("cncrt", frame_masked)
                 
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        _, frame_t = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", frame_t)
              
        white_px = np.argwhere(frame_t)
        unique, counts = np.unique(white_px[:,1], return_counts=True)
        counts = -counts + 2*np.abs(unique - w/2)
        arg_maxes = np.argsort(counts, 0)
        desired_maxes = 10
        sorted_white_px_bool = np.isin(white_px[:, 1], unique[arg_maxes[:desired_maxes]])
        sorted_white_px = white_px[sorted_white_px_bool]
        
        h, w = frame_gray.shape
        right_sorted_white_px_bool = sorted_white_px[:,1] > .5*w
        left_sorted_white_px_bool = np.invert(right_sorted_white_px_bool)
        
        right_sorted_white_px = sorted_white_px[right_sorted_white_px_bool]
        left_sorted_white_px = sorted_white_px[left_sorted_white_px_bool]
        
        frame[right_sorted_white_px[:,0], right_sorted_white_px[:,1]] = [0,0,255]
        frame[left_sorted_white_px[:,0], left_sorted_white_px[:,1]] = [255,0,0]
        cv2.imshow("lane", frame)
        
        poly_ft = PolynomialFeatures(2)
        all_y = np.array(range(h))
        X_all_y = poly_ft.fit_transform(all_y.reshape(-1,1))
        
        right_avg_x = 1000
        left_avg_x = 1000
        
        try:
            right_poly = poly_ft.fit_transform(right_sorted_white_px[:, 0].reshape(-1,1))
            right_fit = RANSACRegressor(residual_threshold=2).fit(right_poly, right_sorted_white_px[:, 1])
            right_predict = right_fit.predict(X_all_y)
            right_predict = np.round(right_predict).astype(int, copy=False)
            right_avg_x = np.mean(right_predict)
        except ValueError:
            pass

        try:
            left_poly = poly_ft.fit_transform(left_sorted_white_px[:, 0].reshape(-1,1))
            left_fit = RANSACRegressor(residual_threshold=2).fit(left_poly, left_sorted_white_px[:, 1])
            left_predict = left_fit.predict(X_all_y)
            left_predict = np.round(left_predict).astype(int, copy=False)
            left_avg_x = np.mean(left_predict)
        except ValueError:
            pass
        
        dist = abs(left_avg_x - right_avg_x)
        if self.approx_perp_dist == None:
            self.approx_perp_dist = dist
        
        if abs(dist - self.approx_perp_dist) < 50:    
            left_worked = False
            right_worked = False
            
            try:
                frame[all_y, right_predict] = [0,0,255]
                right_worked = True
            except IndexError:
                pass
            
            try:
                frame[all_y, left_predict] = [255,0,0]
                left_worked = True
            except IndexError:
                pass
            
            if left_worked and right_worked:
                left_right = np.vstack((left_predict, right_predict))
                center_predict = np.mean(left_right, axis=0).astype(np.uint8, copy=False)
                
                prev_arrow = False
                arr_res = 10
                
                
                inv_M = np.linalg.pinv(M)
                start = 5
                for arrow_ind in range(10):
                    if start + arr_res >= h:
                        break
                    if not prev_arrow:
                        pts = cv2.perspectiveTransform(np.array([[[center_predict[start+arr_res], start+arr_res]],
                                                                 [[center_predict[start], start]]], dtype=np.float32), inv_M)
                        pts = np.array(np.round(pts), dtype=int)
                        # frame = cv2.UMat(frame)
                        # cv2.arrowedLine(frame, (center_predict[arrow_ind+arr_res], arrow_ind+arr_res),
                        #                        (center_predict[arrow_ind], arrow_ind), (0,255,0), 2)
                        # frame = cv2.UMat(frame)

                        cv2.arrowedLine(orig_frame, tuple(np.round(pts[1][0])), tuple(np.round(pts[0][0])), (0,255,0), 5, tipLength=.5)
                        start += arr_res
                        arr_res += 10
                    else:
                        start += 10
                    prev_arrow = not prev_arrow
                    
                    
                    
                
                # frame[all_y, center_predict] = (0,255,0)
            
                
            
        return orig_frame


if __name__ == "__main__":
    system('cls')
    
    vid_name = 'data_1/data_vid1.avi'
    D = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                  [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    # selected_corners = [(348, 173), (450, 173), (520, 247), (190, 247)]
    selected_corners =  [(393, 154), (425, 154), (486, 246), (219, 246)]
    
    # vid_name = 'data_2/challenge_video.mp4'
    # D = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])
    # K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
    #               [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
    #               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # selected_corners = [(333, 272), (424, 272), (529, 337), (229, 337)]
    
    # This function takes all of the images and patches them into a video. It then saves the resulting video. Therefore,
    # this function must only be run if the video has not been generated yet 
    # VideoHandler.gen_video_frm_imgs(vid_name, 'data_1/data', 10, 302, '.png', 10)
    
    lane_detector = LaneDetection()
    handle = LaneDetectionHandler(vid_name, 50, K, D)
    
    # These two lines bring up a prompt that allows the user to pick coordinates for the corners
    # handle.get_lane_corners()
    
    
    handle.set_projective_transform(selected_corners)
    
    def body(frame):
        frame = lane_detector.detect_lanes(frame)
        cv2.imshow("img", frame)
        
    handle.run(body)
    
    
    
    
    
    
    
    
    