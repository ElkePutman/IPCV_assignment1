import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self, input_file: str, output_file: str, down_fact: float = 1.0):
        self.input_file = input_file
        self.output_file = output_file
        self.cap = cv2.VideoCapture(input_file)
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.down_fact = down_fact
        self.new_width = int(self.width * down_fact)
        self.new_height = int(self.height * down_fact)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.new_width, self.new_height))
        self.frame = None 
        self.current_time = None
        self.write_frame = True
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_time_vid = ((self.frame_count - 1) / self.fps) * 1000

        print('Processing new video')


    def put_text(self,text, x=50, y=50, color=(255, 0, 0),sz_in = 0.7,inp=None,th_in = 2):

        sz = self.down_fact * sz_in
        th = int(self.down_fact * th_in)
        x_pos = int(self.down_fact * x)
        y_pos = int(self.down_fact * y)
        if inp is None:
            cv2.putText(self.frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, color, th, cv2.LINE_AA)
        else:
            cv2.putText(inp, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, color, th, cv2.LINE_AA)

    # helper function to change what you do based on video seconds
    def between(self, lower=None, upper=None) -> bool:
        if lower is not None and upper is not None:
            return lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
        else:
            return self.lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < self.upper

    # downsample the video to a lower resolution
    def downsample(self):
        if (self.new_width, self.new_height) != (self.width, self.height):
            self.frame = cv2.resize(self.frame, (self.new_width, self.new_height))

    # Make the frames gray
    def to_gray(self):
        start_time = 1000               
        if not start_time <= self.current_time <= self.end_time_vid:
            return

        if self.frame.shape[2] == 1:
            return
        else:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            if 0 <= self.current_time < 1999:
                self.put_text("Color to Grey using cvtColor")

    # apply gamma transform based on a lineair equation
    def gamma_transform(self, start_time, duration):
        gamma_0 = 0
        gamma_end = 5 
        end_time = start_time + duration - 1
        shifted_time = self.current_time - start_time
        gamma = ((gamma_end - gamma_0) / duration) * shifted_time + gamma_0

        if not start_time <= self.current_time <= end_time:
            return
        else:
            self.frame = np.array(np.max(self.frame) * (self.frame / np.max(self.frame)) ** gamma, dtype='uint8')
            self.put_text(f"Gamma={gamma:.2f}")

    # add smoothing with a box filter
    def smoothing(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        kernel = (1/9) * np.ones((3, 3))
        self.frame = cv2.filter2D(self.frame, -1, kernel)
        self.put_text("Smoothing with box filter")

    # using a sharpening filter
    def sharpening(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.frame = cv2.filter2D(self.frame, -1, kernel)
        self.put_text("Sharpening with sharpening filter")

    # apply custom filter for edge detection
    def custom1(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        self.frame = cv2.filter2D(self.frame, -1, kernel)
        self.put_text("Vertical edge detection")

    # apply custom filter 2
    def custom2(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
        self.frame = cv2.filter2D(self.frame, -1, kernel)
        self.put_text("Custom filter 2")

    # Apply fourier transform to the frame
    def fourier(self, start_time, duration, return_spectrum=False):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) #go back to one channel for fourier functions
        self.frame = self.frame / np.max(self.frame) 
        IM = np.fft.fft2(self.frame)
        IM = np.fft.fftshift(IM)
        if return_spectrum:
            return IM
        else:
            IMmag = np.abs(IM)
            IMlog = np.log(IMmag + 10)
            IMlog = cv2.normalize(IMlog, None, 0, 255, cv2.NORM_MINMAX)
            fft_corr = np.uint8(IMlog)
            self.write_frame = False
            self.put_text("DFT spectrum",inp=fft_corr)
            
            return fft_corr

    # create a 2D gaussian for the fourier filters
    def gaussian(self, D0, type='low'):
        r, c = self.frame.shape[:2]
        U, V = np.meshgrid(np.arange(r), np.arange(c), indexing='ij')
        D = np.sqrt((U - r/2)**2 + (V - c/2)**2)
        if type == 'low':
            H = np.exp(-(D**2) / (2.0 * D0**2))
        elif type == 'high':
            H = 1.0 - np.exp(-(D**2) / (2.0 * D0**2))
        else:
            raise ValueError("No valid filter type")
        return H

    # apply a low pass filter
    def low_pass(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        spectrum = self.fourier(start_time, duration, return_spectrum=True)
        D0 = 20
        tf = self.gaussian(D0, type='low')
        filtered = spectrum * tf
        ifft2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
        ifft2 = cv2.normalize(ifft2, None, 0, 255, cv2.NORM_MINMAX)
        self.frame = np.uint8(ifft2)
        self.write_frame = True
        self.put_text(f"Low pass filter with sigma ={D0:.1f}")

    # apply a high pass filter
    def high_pass(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        spectrum = self.fourier(start_time, duration, return_spectrum=True)
        D0 = 20
        tf = self.gaussian(D0, type='high')
        filtered = spectrum * tf
        ifft2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
        ifft2 = cv2.normalize(ifft2, None, 0, 255, cv2.NORM_MINMAX)
        self.frame = np.uint8(ifft2)
        self.write_frame = True
        self.put_text(f"High pass filter with sigma ={D0:.1f}")

    # apply a band pass filter
    def band_pass(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        spectrum = self.fourier(start_time, duration, return_spectrum=True)
        D0_in, D0_out = 40, 30
        inner = self.gaussian(D0_in, type='low')
        outer = self.gaussian(D0_out, type='low')
        tf = outer - inner
        filtered = spectrum * tf
        ifft2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
        ifft2 = cv2.normalize(ifft2, None, 0, 255, cv2.NORM_MINMAX)
        self.frame = np.uint8(ifft2)
        self.write_frame = True
        self.put_text(f"Band pass filter with sigma_1 ={D0_in:.1f} and sigma_2 ={D0_out:.2f}",sz_in=0.4, th_in=1)

    # apply binary thresholding
    def thresholding(self, start_time, duration, threshold_value=100):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= self.end_time_vid:
            return
        ret, binary_image = cv2.threshold(self.frame, threshold_value, np.max(self.frame), cv2.THRESH_BINARY)
        self.frame = binary_image
        if start_time <= self.current_time <= end_time:
            self.put_text(f"Binary thresholding with threshold = {threshold_value}")

    # apply binary opening
    def opening(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        self.frame = (morphology.binary_opening(self.frame).astype(np.uint8)) * 255
        self.put_text("Opening")

    # apply binary closing
    def closing(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        self.frame = (morphology.binary_closing(self.frame).astype(np.uint8)) * 255
        self.put_text("Closing")

    # apply binary dilation
    def dilation(self, start_time, duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        self.frame = (morphology.binary_dilation(self.frame).astype(np.uint8)) * 255
        self.put_text("Dilation")

    # apply binary erosion
    def erosion(self, start_time, duration):
        end_time =self.end_time_vid
        # end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        self.frame = (morphology.binary_erosion(self.frame).astype(np.uint8)) * 255
        self.put_text("Erosion")


    # run all the funtions
    def run(self, show_video=False):
        #list of filters (funciton,duration(ms))
        filters = [
            (self.gamma_transform, 6000),
            (self.smoothing, 3000),
            (self.sharpening, 3000),
            (self.custom1, 3000),
            (self.custom2, 3000),
            (self.fourier, 5000),
            (self.low_pass, 5000),
            (self.high_pass, 5000),
            (self.band_pass, 5000),
            (self.thresholding, 4000),
            (self.opening, 4000),
            (self.closing, 4000),
            (self.dilation, 4000),
            (self.erosion, 4000),
        ]

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.downsample()
            self.to_gray()

            start = 2000
            fft_frame = None 

            for func, dur in filters:

                result = func(start, dur)
                if func == self.fourier:
                    fft_frame = result
                start += dur

            if self.write_frame:
                self.out.write(self.frame)
            else:
                self.out.write(fft_frame)
                self.write_frame = True

            if show_video:
                cv2.imshow('Video', self.frame)
            # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break


    # if debugging is needed, run this
    def debug_single_frame(self, timestamp_ms, show_video=True, save_frame=False):

        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Couldn't read the frame on timestamp", timestamp_ms)
            return
        
        self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        print(f"Debug frame on {self.current_time} ms")

        self.downsample()
        self.to_gray()

    
        filters = [
            (self.gamma_transform, 6000),
            (self.smoothing, 3000),
            (self.sharpening, 3000),
            (self.custom1, 3000),
            (self.custom2, 3000),
            (self.fourier, 5000),
            (self.low_pass, 5000),
            (self.high_pass, 5000),
            (self.band_pass, 5000),
            (self.thresholding, 4000),
            (self.opening, 4000),
            (self.closing, 4000),
            (self.dilation, 4000),
            (self.erosion, 4000),
        ]

        start = 2000
        fft_frame = None

        for func, dur in filters:
            result = func(start, dur)
            if func == self.fourier:
                fft_frame = result
            start += dur

        frame_to_show = self.frame if self.write_frame else fft_frame
        self.write_frame = True

        if show_video:
            cv2.imshow('Debug Frame', frame_to_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_frame:
            cv2.imwrite("debug_frame.png", frame_to_show)
            print("Saved frame")
            im = cv2.imread('debug_frame.png')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(im)
            plt.show()   
 
