# Tennis_Analysis

## Description
The Tennis Analysis System utilizes computer vision and machine learning to analyze tennis match footage. Given a television view video clip of a tennis rally, this system provides detailed analysis including player and ball detection with bounding boxes and key-court points as well, overlay of a mini-court displaying player and ball positions, calculation of player and ball speeds adjusted for frame rate, statistics on shots played by each player, and generation of heatmaps based on player movements.

## Output
https://github.com/VasuVerma990/Tennis_Analysis/assets/131004786/a2c55ae8-e154-41a6-ae48-2044937cc37f

## Features
- Player and Ball Detection: Detects players and the tennis ball in each frame of the input video, overlaying bounding boxes for visualization.
- Key-Court Points and Mini-Court and players and ball speed Overlay: Identifies key points on the tennis court (baseline, service lines, etc.) and overlays a 
  mini-court on the video displaying player and ball positions, and average speed of players and ball .
- Shot Analysis: Counts the number of shots played by each player throughout the rally, providing statistical insights into shot types and frequencies.
- Dataframe Output: Generates a structured DataFrame capturing players and ball speeds at each frame, enabling detailed frame-by-frame analysis.
- Heatmap Generation: Creates heatmaps of player movements on the court based on their bounding boxes, visualizing player activity and positioning trends.
  
##  Requirements
- python3.8
- ultralytics
- pytroch
- pandas
- numpy
- opencv
## Direction of Use
- Setting Up the Video Path:
   - In the main.py file, specify the path to the video you want to analyze.
   - Ensure the directory structure allows for function calls and data storage.
- Running the Analysis:
   - When you run the main.py file for the first time, it will generate pickle files for player detection, ball detection, and key-court points.
   - These pickle files will be saved and can be reused for the same video without needing to run the entire analysis again.
- Reusing Pickle Files:
   - To reuse the generated pickle files, ensure they are in the appropriate directory.
   - The pickle files can also be used in the ball_analysis.ipynb and player_analysis.ipynb notebooks for further analysis, such as shot counts, ball hits           surface (under assumptions i.e., difference between the player hit the ball and the surface is 30 frames), and player heatmap generation.
- Tweaking Key-Court Points:
   - If the correct players are not detected, you may need to tweak the key-court points to be nearest to the players.
   - Adjustments can be made in the respective functions within the code to improve detection accuracy.
## Models Used
- YOLO v8 for player detection
- Fine Tuned YOLO v8x for tennis ball detection
- Court Key point extraction using Resnet50
- Model trained : https://drive.google.com/drive/folders/1Mvk682ya48Xy38X5FC5O0pWY7prgpfdj


