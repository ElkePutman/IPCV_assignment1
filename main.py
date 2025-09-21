import argparse
import sys
import os

from Processor_class import VideoProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='input video filename (zonder path)')
    parser.add_argument('-o', "--output", help='output video filename (zonder path)')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide input and output video filenames! See --help")

    # zet hier je standaardpad
    BASE_INPUT_PATH = r"C:\Users\ejput\OneDrive - University of Twente\BME\252601-Kwartiel 1 2025\Image processing\Assignment_1"
    BASE_OUTPUT_PATH = r"C:\Users\ejput\OneDrive - University of Twente\BME\252601-Kwartiel 1 2025\Image processing\Assignment_1\Processed_videos"

    input_file = os.path.join(BASE_INPUT_PATH, args.input)
    output_file = os.path.join(BASE_OUTPUT_PATH, args.output)

    process = VideoProcessor(input_file, output_file,down_fact=0.5)

    process.run(show_video=False)
    # process.debug_single_frame(57000, show_video=False,save_frame=True)