import cv2

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

        print('initializing new video')


    # helper function to change what you do based on video seconds
    def between(self, lower: int, upper: int) -> bool:
        return lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

    #downsample the video to a lower resolution
    def downsample(self, frame):
        if (self.new_width, self.new_height) != (self.width, self.height):
            frame = cv2.resize(frame, (self.new_width, self.new_height))
        return frame
    
    def to_gray(self):
        if(self.frame.shape[2]==1):
            return
        else:
            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)  
        return
        
        
    
    
    def intensity_transform(self,frame):
        return frame


    
    def run(self, show_video=False):

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret: #ret is true if frame is correct gelezen
                break
            
            # if cv2.waitKey(28) & 0xFF == ord('q'):
            #     break
            if self.between(1000, 39999):
                self.to_gray()                
                
            # ...

            # write frame that you processed to output
            self.out.write(self.frame)

            # (optional) display the resulting frame
            if show_video:
                cv2.imshow('Video', self.frame)

            # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



    
