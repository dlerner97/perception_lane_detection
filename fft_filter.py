from os import stat
import cv2
import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

# FFT Class
class Background_FFT:
    def __init__(self):
        pass

    # Resize the frame
    @staticmethod
    def resize_frame(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # Separate background with fft
    @staticmethod
    def separate_background_w_fft(frame, filter_size=50):
        # def disp_fft(img):
        #     plt.figure()
        #     plt.imshow(np.log(1+np.abs(img)).real, "gray") 

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Take fourier transform and shift to the center
        im_fft = fftpack.fft2(frame)
        im_fft_shift = fftpack.fftshift(im_fft)

        # Apply a high pass mask/filter fft
        r, c = im_fft_shift.shape
        mask = np.zeros_like(frame)
        cv2.circle(mask, (int(c/2), int(r/2)), filter_size, 255, -1)[0]
        high_pass = im_fft_shift * np.invert(mask)/255

        # Invert fft
        im_fft_ishift = fftpack.ifftshift(high_pass)
        ifft = np.abs(fftpack.ifft2(im_fft_ishift))

        # Convert to grayscale
        min_val = np.min(ifft)
        max_val = np.max(ifft)
        gray2uint8 = lambda px: np.uint8(255*(px - min_val)/(max_val-min_val))
        
        # Post process
        mat = np.asarray(list(map(gray2uint8, ifft)))
        # _,mat = cv2.threshold(mat, 150, 255, cv2.THRESH_BINARY)
        # mat = cv2.dilate(mat, (57,57))
        return mat

if __name__ == '__main__':
    from os import system
    system('cls')

    # Run FFT for a single frame
    fft = Background_FFT('Tag0.mp4')
    _, frame = fft.video_feed.read()
    img = fft.separate_background_w_fft(frame)
    cv2.imwrite("fft_background.png", img)

    cv2.imshow("frame", img)
    cv2.waitKey(0)



