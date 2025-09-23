Video Processing Script
=======================

This project is Assignment 1 from the Image processing & computer vision course
The functions are implemented in the 'VideoProcessor' class (defined in 'Processor_class.py').  
The script takes an input video and produces a processed output video.  

-----------------------------------
Requirements
-----------------------------------
- Python 3.8 or higher
- Packages:
  - opencv-python
  - numpy
  - scikit-image
  - matplotlib
 
-----------------------------------
Usage
-----------------------------------
Run the script from the command line:

    python main.py -i <input_video> -o <output_video>

Arguments:
- -i / --input   : Input video filename (without path)
- -o / --output  : Output video filename (without path)

IMPORTANT: Set Your Own Paths!
-----------------------------------
In the script ('main.py') the input and output base paths are defined as:

    BASE_INPUT_PATH  = r"C:\Users\...\Assignment_1"
    BASE_OUTPUT_PATH = r"C:\Users\...\Assignment_1\Processed_videos"

You MUST change these paths to match your own folder structure before running the script.  
Otherwise, the program will not find your videos or save the output correctly.

-----------------------------------
Example
-----------------------------------
Suppose you have a file called 'example.mp4' in the input folder:

    python main.py -i example.mp4 -o example_processed.mp4

This will:
- Read from:
  C:\Users\<username>\OneDrive - University of Twente\BME\252601-Kwartiel 1 2025\Image processing\Assignment_1\example.mp4
- Write to:
  C:\Users\<username>\OneDrive - University of Twente\BME\252601-Kwartiel 1 2025\Image processing\Assignment_1\Processed_videos\example_processed.mp4

-----------------------------------
Options in the Code
-----------------------------------
- down_fact (in VideoProcessor)  
  Controls downsampling of the video. Default = 1 (no downsampling).

- process.run(show_video=False)  
  If show_video=True, the processed video will be displayed during execution.  
  If show_video=False, the result will only be saved.

- process.debug_single_frame(timestamp_ms, show_video=False, save_frame=True)  
  Allows debugging of a single frame (e.g. to check intermediate processing results).

-----------------------------------
Notes
-----------------------------------
- Make sure the input video file exists in the input folder.
- You can edit 'BASE_INPUT_PATH' and 'BASE_OUTPUT_PATH' in the code to match your own setup.
