import cv2
import numpy as np

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

        print('initializing new video')


    # helper function to change what you do based on video seconds
    def between(self, lower = None, upper=None) -> bool:
        if lower is not None and upper is not None:
            return lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
        else:
            return self.lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < self.upper

    #downsample the video to a lower resolution
    def downsample(self, frame):
        if (self.new_width, self.new_height) != (self.width, self.height):
            frame = cv2.resize(frame, (self.new_width, self.new_height))
        return frame
    
    def to_gray(self):
        # if not self.between(1000, 39999):
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

 

        
        
    
    



    
    def run(self, show_video=False):

        

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret: #ret is true if frame is correct gelezen
                break
            
            # if cv2.waitKey(28) & 0xFF == ord('q'):
            #     break
            self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
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
            # if self.between(1000, 39999):
            #     self.to_gray()
                
            # if self.between(2000,7999):

            #     self.gamma_transform()                
                
            # ...

            # write frame that you processed to output
            self.out.write(self.frame)

            # (optional) display the resulting frame
            if show_video:
                cv2.imshow('Video', self.frame)

            # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



    
