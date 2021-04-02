import cv2
import numpy as np
from os import system
from video_handler import VideoHandler
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

class LaneDetectionHandler(VideoHandler):
    def __init__(self, video_name, video_out_name, scale_percent, K, D, lane_output_shape=None) -> None:
        super().__init__(video_name, video_out_name, scale_percent=scale_percent)
        self.K = K
        self.D = D
        h,w = super().get_img_shape()
        
        if lane_output_shape == None:
            self.lane_output_shape = (int(.5*w),int(2.5*h/2))
        elif isinstance(lane_output_shape, tuple):
            self.lane_output_shape = (int(lane_output_shape[0]*w),int(lane_output_shape[1]*h))
        else:
            self.lane_output_shape = lane_output_shape
            
        self.ideal_cam_mat, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D,(w,h), 1, (w,h))
        
    def set_projective_transform(self, selected_corners, bounds_x, bounds_y):
        h, w = self.lane_output_shape 
        
        bounds_x = (int(bounds_x[0]*w), int(bounds_x[1]*w))
        bounds_y = (int(bounds_y[0]*h), int(bounds_y[1]*h))
                
        selected_corners = np.float32(selected_corners)
        birds_eye_corners = np.float32([[bounds_x[0],bounds_y[1]], [bounds_x[1], bounds_y[1]], [bounds_x[1], bounds_y[0]], [bounds_x[0], bounds_y[0]]])
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
            # frame = self.resize_frame(frame, 50)
        except TypeError:
            pass
        finally:
            return ret, (frame, frame_cp, self.M)
    
    def get_lane_corners(self, scale=200, n=0):
        print("Please select four corners for homography in the clockwise direction.")
        print("Use W/A/S/D to move the circle and set the four corners.")
        print("Use E/R to lower/upper the pixel movement per key press.")
        print("Select V to set the image to a new image and x to reset selection.")
        print("Lastly select Q to quit. If four corners are chosen, the program will print the selected corners.")
        
        img = self.resize_frame(self.get_frame_n(n), scale)        
        h, w = img.shape[:2]
        circle_x = w//2
        circle_y = h//2
        res = 5
        corners = []
        pt_scale = scale/100
        
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
                    img = self.resize_frame(self.get_frame_n(n), scale) 
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
            
            corners = [(int(round(corner[0]/pt_scale)), int(round(corner[1]/pt_scale))) for corner in corners]    
            print(f"The four corners in clockwise are: {corners}")
            return corners
        except KeyboardInterrupt:
            return

class LaneDetection:
    def __init__(self, desired_maxes = 50, threshold=220, mid_per=.6, grass=False) -> None:
        self.approx_perp_dist = None
        self.desired_maxes = desired_maxes
        self.threshold = threshold
        self.mid_per = mid_per
        self.grass = grass
    
    def detect_lanes(self, frame_info):
        M = frame_info[-1]
        orig_frame = frame_info[1]
        frame = frame_info[0]
        
        h, w = frame.shape[:2]                   
        
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_t_white = cv2.inRange(frame_hsv, (0, 0, self.threshold), (255, 255, 255))
        frame_t = None
        
        if not self.grass:
            frame_t_yellow = cv2.inRange(frame_hsv, (10, 50, 50), (50,255,200))
            frame_t = cv2.bitwise_or(frame_t_white, frame_t_yellow)
        else:
            frame_t = frame_t_white

        cv2.imshow("frmae", frame)
        cv2.imshow("thresh", frame_t)
        # cv2.waitKey(0)
              
        white_px = np.argwhere(frame_t)
        unique, counts = np.unique(white_px[:,1], return_counts=True)
        arg_maxes = np.argsort(-counts, 0)
        sorted_white_px_bool = np.isin(white_px[:, 1], unique[arg_maxes[:self.desired_maxes]])
        sorted_white_px = white_px[sorted_white_px_bool]
        
        h, w = frame_t.shape
        right_sorted_white_px_bool = sorted_white_px[:,1] > self.mid_per*w
        left_sorted_white_px_bool = np.invert(right_sorted_white_px_bool)
        
        right_sorted_white_px = sorted_white_px[right_sorted_white_px_bool]
        left_sorted_white_px = sorted_white_px[left_sorted_white_px_bool]
        
        frame[right_sorted_white_px[:,0], right_sorted_white_px[:,1]] = [0,0,255]
        frame[left_sorted_white_px[:,0], left_sorted_white_px[:,1]] = [255,0,0]
        
        poly_ft = PolynomialFeatures(2)
        all_y = np.array(range(h))
        X_all_y = poly_ft.fit_transform(all_y.reshape(-1,1))
        
        right_avg_x = 5000
        left_avg_x = 5000
        
        try:
            right_poly = poly_ft.fit_transform(right_sorted_white_px[:, 0].reshape(-1,1))
            right_fit = RANSACRegressor(residual_threshold=5).fit(right_poly, right_sorted_white_px[:, 1])
            right_predict = right_fit.predict(X_all_y)
            right_predict = np.round(right_predict).astype(int, copy=False)
            right_avg_x = np.mean(right_predict)
        except ValueError:
            pass

        try:
            left_poly = poly_ft.fit_transform(left_sorted_white_px[:, 0].reshape(-1,1))
            left_fit = RANSACRegressor(residual_threshold=5).fit(left_poly, left_sorted_white_px[:, 1])
            left_predict = left_fit.predict(X_all_y)
            left_predict = np.round(left_predict).astype(int, copy=False)
            left_avg_x = np.mean(left_predict)
        except ValueError:
            pass
        
        dist = abs(left_avg_x - right_avg_x)
        if self.approx_perp_dist == None:
            self.approx_perp_dist = dist
            
        # cv2.waitKey(0)
        
        text = "Cannot Detect Lanes"
        text_x = w//2-170
        if abs(dist - self.approx_perp_dist) < 80:    
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
                center_predict = np.mean(left_right, axis=0).astype(np.uint16, copy=False)
                
                text_x = w//2-40
                curve = int(center_predict[0]) - int(center_predict[h//2])
                if abs(curve) < 20:
                    text = "Going straight"
                elif curve > 0:
                    text = "Turning right"
                else:
                    text = "Turning left"
                
                prev_arrow = False
                arr_res = 10
                
                inv_M = np.linalg.pinv(M)
                start = 5
                
                overlay = np.zeros_like(frame)
                frame = cv2.UMat(frame)
                                
                partial_h = h//4
                pts_ls = [[[left_predict[0], 0],
                           [left_predict[partial_h], partial_h],
                           [left_predict[h-1], h-1],
                           [right_predict[h-1], h-1],
                           [right_predict[partial_h], partial_h],
                           [right_predict[0], 0]]]
                
                pts = np.array(pts_ls, dtype=np.int32)
                               
                cv2.fillPoly(overlay, np.array(pts_ls, dtype=np.int32), (255,255,0))
                frame = cv2.addWeighted(overlay, .2, frame, 1 - .2, 0)
                
                overlay2 = np.zeros_like(orig_frame)
                pts = cv2.perspectiveTransform(np.array(pts_ls, dtype=np.float32), inv_M).astype(np.int32, copy=False)
                cv2.fillPoly(overlay2, pts, (255,255,0))
                orig_frame = cv2.addWeighted(overlay2, .2, orig_frame, 1 - .2, 0)
                
                for arrow_ind in range(20):
                    if start + arr_res >= h:
                        break
                    if not prev_arrow:
                        cv2.arrowedLine(frame, (center_predict[start+arr_res], start+arr_res), (center_predict[start], start), (0,255,0), 5, tipLength=.5)
                        pts_c = cv2.perspectiveTransform(np.array([[[center_predict[start+arr_res], start+arr_res]],
                                                                   [[center_predict[start], start]]], dtype=np.float32), inv_M)
                        pts_c = np.array(np.round(pts_c), dtype=int)
                                                
                        cv2.arrowedLine(orig_frame, tuple(np.round(pts_c[1][0])), tuple(np.round(pts_c[0][0])), (0,255,0), 5, tipLength=.5)
                        start += arr_res
                        arr_res += 10
                    else:
                        start += 10
                    prev_arrow = not prev_arrow            
        
        
        cv2.putText(orig_frame, text, (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 0, 150), 2)        
        cv2.imshow("lane", frame)
        return orig_frame


if __name__ == "__main__":
    system('cls')
    
    vid_name = 'data_1/data_vid1.avi'
    video_out_name = 'data_vid1_lanes'
    D = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                  [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    selected_corners = [(647, 246), (692, 246), (798, 421), (388, 421)]
    lane_output_shape = (.6, 2.5/2)
    bounds_x = (.5, .95)
    bounds_y = (0, 1.5) 
    desired_maxes = 80
    threshold = 220
    mid_per = .6
    grass = True
    
    # vid_name = 'data_2/challenge_video.mp4'
    # video_out_name = 'challenge_video_lanes'
    # D = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])
    # K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
    #               [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
    #               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # selected_corners = [(657, 455), (713, 455), (963, 645), (381, 645)]
    # lane_output_shape = (.3, .6)
    # bounds_x = (.1, .50)
    # bounds_y = (0, 1)
    # desired_maxes = 140
    # threshold = 175
    # mid_per = .4
    # grass = False
    
    # This function takes all of the images and patches them into a video. It then saves the resulting video. Therefore,
    # this function must only be run if the video has not been generated yet 
    # VideoHandler.gen_video_frm_imgs(vid_name, 'data_1/data', 10, 302, '.png', 10)
    
    lane_detector = LaneDetection(desired_maxes, threshold, mid_per, grass)
    handle = LaneDetectionHandler(vid_name, video_out_name, 100, K, D, lane_output_shape=lane_output_shape)
    
    # These two lines bring up a prompt that allows the user to pick coordinates for the corners
    # handle.get_lane_corners(scale=100)
    
    
    handle.set_projective_transform(selected_corners, bounds_x, bounds_y)
    
    def body(frame):
        frame = lane_detector.detect_lanes(frame)
        cv2.imshow("img", frame)
        return frame
        
    handle.run(body)
    
    
    
    
    
    
    
    
    