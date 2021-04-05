# Perception: Lane Detection

Lane detection and turn prediction.

## Run Instructions

The lane_detection.py script handles multiple functions. Firstly, it runs the computer vision and detection algorithms for the two sample videos. Second, the script has functionality for enabling the user to choose custom corners for each video. These corners will be used in the homography step. Lastly, the script also has a function for generating the data1 video from the given images. See how to run these scripts below.

Please follow the terminal instructions once a program is run.

To run the lane detection script, run
```bash
lane_detection.py
```
with the following command line arguments:

1. Run lane detection on the non-challenge video:
```bash
python lane_detection.py
```
2. Run lane detection on challenge video:
```bash
python lane_detection.py challenge
```
3. Execute manual corner finder on non-challenge video:
```bash
python lane_detection.py def_lane_corners
```
4. Execute manual corner finder on challenge video:
```bash
python lane_detection.py challenge def_lane_corners
```
5. Generate non-challenge video from its images:
```bash
python lane_detection.py gen_video
```
