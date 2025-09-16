import cv2
import numpy as np
from scipy.fft import ifftshift, fftshift, fftfreq, fft2
import imageio
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self, input_file: str, output_file: str, new_width=None, new_height=None):
        self.input_file = input_file
        self.output_file = output_file
        self.cap = cv2.VideoCapture(input_file)
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_width = new_width or self.width
        self.new_height = new_height or self.height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.new_width, self.new_height))
        self.frame=None 
        self.current_time = None
        self.write_frame = True

        print('Processing new video')


    # helper function to change what you do based on video seconds
    def between(self, lower = None, upper=None) -> bool:
        if lower is not None and upper is not None:
            return lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
        else:
            return self.lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < self.upper

    #downsample the video to a lower resolution
    def downsample(self):
        if (self.new_width, self.new_height) != (self.width, self.height):
            self.frame = cv2.resize(self.frame, (self.new_width, self.new_height))
        return 
     
    def to_gray(self):
        start_time = 1000       
        end_time = 39999
        if not start_time<=self.current_time<=end_time:
            return

        if(self.frame.shape[2]==1):
            return
        else:
            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)  
            return
    
    def gamma_transform(self,start_time,duration):
        gamma_0 = 0
        gamma_end = 5 
        # start_time = 2000        
        end_time = start_time+duration-1
               
        shifted_time = self.current_time - start_time #start at t =0
        gamma = ((gamma_end-gamma_0)/duration)*shifted_time +gamma_0
              

        if not start_time<=self.current_time<=end_time:
            return
        else:
            self.frame = np.array(255*(self.frame / np.max(self.frame)) ** gamma, dtype = 'uint8')
  
            return
    def smoothing(self,start_time,duration):
        end_time = start_time+duration-1
        if not start_time<=self.current_time<=end_time:
            return
        else:
            kernel = (1/9)*np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]])
            self.frame = cv2.filter2D(self.frame, -1, kernel)  
            return
        
    def sharpening(self,start_time,duration):
        end_time = start_time+duration-1
        if not start_time<=self.current_time<=end_time:
            return
        else:
            kernel = np.array([[0,-1,0],
                     [-1,5,-1],
                     [0,-1,0]])
            self.frame = cv2.filter2D(self.frame, -1, kernel)  
            return
        
        
    def custom1(self,start_time,duration):
        end_time = start_time+duration-1
        if not start_time<=self.current_time<=end_time:
            return
        else:
            kernel = np.array([[1,0,-1],
                     [1,0,-1],
                     [1,0,-1]])
            self.frame = cv2.filter2D(self.frame, -1, kernel)  
            return 
           
    def custom2(self,start_time,duration):
        end_time = start_time+duration-1
        if not start_time<=self.current_time<=end_time:
            return
        else:
            kernel = np.array([[0,1,0],
                     [1,1,1],
                     [0,1,0]])
            self.frame = cv2.filter2D(self.frame, -1, kernel)  
            return   

    def fourier(self,start_time,duration, return_spectrum=False):
        end_time = start_time+duration-1        
        if not start_time<=self.current_time<=end_time:
            return
        else:
            # fft  = fftshift(fft2(ifftshift(self.frame)))
            # fft_mag = np.abs(fft)
            # fft_db = 20 * np.log10(fft_mag + 1e-8)
            # # fft_norm = cv2.normalize(fft_db, None, 0, 255, cv2.NORM_MINMAX) #normalize to 0-255
            # fft_corr = np.uint8(fft_db)

            self.frame = self.frame/np.max(self.frame)
            IM = np.fft.fft2(self.frame) 
            IM = np.fft.fftshift(IM) 
            if return_spectrum:
                return IM
            else:  
                IMmag = np.abs(IM) 
                IMlog = np.log(IMmag+10) # logarithmic 
                IMlog = cv2.normalize(IMlog,None,0,255, cv2.NORM_MINMAX)
                fft_corr = np.uint8(IMlog)
                # print(np.max(fft_corr))
                self.write_frame = False
                return  fft_corr
        
    # def low_pass(self,start_time,duration):
    #     end_time = start_time+duration-1
    #     if not start_time<=self.current_time<=end_time:
    #         return
    #     fft = self.fourier(start_time,duration)
    #     #Create a low pass filter image
    #     x,y = fft.shape[1],fft.shape[0]
    #     #size of circle
    #     e_x,e_y=500,500
    #     #create a box 
    #     bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

    #     low_pass_filt=Image.new("L",(fft.shape[1],fft.shape[0]),color=0)

    #     draw1=ImageDraw.Draw(low_pass_filt)
    #     draw1.ellipse(bbox, fill=1)

    #     low_pass_np=np.array(low_pass_filt)

    #     # plt.figure()
    #     # plt.imshow(low_pass_filt, cmap="gray")
    #     # plt.title("Low-pass filter result")
    #     # plt.show()

    #     #multiply both the images
    #     fft_norm =10**(fft/20)
    #     filtered=np.multiply(fft_norm,low_pass_np)

    #     #inverse fft
    #     ifft2 = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    #     ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    #     self.frame = np.uint8(ifft2)


    #     self.write_frame = True
    #     return

    def low_pass(self, start_time, duration, r=100):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return

        # pak het complexe spectrum
        spectrum = self.fourier(start_time,duration,return_spectrum=True)

        rows, cols = self.frame.shape[:2]
        crow, ccol = rows // 2, cols // 2

        # circulair masker
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= r*r
        mask = np.zeros((rows, cols), np.uint8)
        mask[mask_area] = 1
        
        # plt.figure()
        # plt.imshow(mask, cmap="gray")
        # plt.title("Low-pass filter result")
        # plt.show()
        # filteren
        filtered = spectrum * mask

        # inverse FFT
        ifft2 = np.fft.ifft2(np.fft.ifftshift(filtered))
        ifft2 = np.real(ifft2)

        # normaliseren naar uint8
        ifft2 = cv2.normalize(ifft2, None, 0, 255, cv2.NORM_MINMAX)
        self.frame = np.uint8(ifft2)
        self.write_frame = True

    def high_pass(self, start_time, duration, r=100):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return

        # pak het complexe spectrum
        spectrum = self.fourier(start_time,duration,return_spectrum=True)

        rows, cols = self.frame.shape[:2]
        crow, ccol = rows // 2, cols // 2

        # circulair masker
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= r*r
        mask = np.ones((rows, cols), np.uint8)
        mask[mask_area] = 0
        
        # plt.figure()
        # plt.imshow(mask, cmap="gray")
        # plt.title("Low-pass filter result")
        # plt.show()
        # filteren
        filtered = spectrum * mask

        # inverse FFT
        ifft2 = np.fft.ifft2(np.fft.ifftshift(filtered))
        ifft2 = np.real(ifft2)

        # normaliseren naar uint8
        ifft2 = cv2.normalize(ifft2, None, 0, 255, cv2.NORM_MINMAX)
        self.frame = np.uint8(ifft2)
        self.write_frame = True

        

   
    def run(self, show_video=False):

        

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret: #ret is true if frame is correct gelezen
                break
            
            # if cv2.waitKey(28) & 0xFF == ord('q'):
            #     break
            self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.downsample()
            self.to_gray()
            start = 2000
            dur = 6000
            self.gamma_transform(start,dur)
            start = start+dur
            dur = 3000
            self.smoothing(start,dur)
            start = start+dur
            dur = 3000
            self.sharpening(start,dur)
            start = start+dur
            dur = 3000
            self.custom1(start,dur)
            start = start+dur
            dur = 3000
            self.custom2(start,dur)
            start = start+dur
            dur = 5000
            fft = self.fourier(start,dur)
            start = start+dur
            dur = 5000
            self.low_pass(start,dur)
            start = start+dur
            dur = 5000
            self.high_pass(start,dur)


            # write frame that you processed to output
            if self.write_frame:
                self.out.write(self.frame)
            else:
                self.out.write(fft)
                self.write_frame = True

            
            if show_video:
                cv2.imshow('Video', self.frame)

            # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



    
