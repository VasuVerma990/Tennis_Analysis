import numpy as np
import cv2

def draw_player_stats(output_video_frames,player_stats,chosen_player):
    id0 = chosen_player[0]
    id1 = chosen_player[1]

    for index, row in player_stats.iterrows():
        player_id0_shot_speed = row[f'player_{id0}_last_shot_speed']
        player_id1_shot_speed = row[f'player_{id1}_last_shot_speed']
        player_id0_speed = row[f'player_{id0}_last_player_speed']
        player_id1_speed = row[f'player_{id1}_last_player_speed']

        avg_player_id0_shot_speed = row[f'player_{id0}_average_shot_speed']
        avg_player_id1_shot_speed = row[f'player_{id1}_average_shot_speed']
        avg_player_id0_speed = row[f'player_{id0}_average_player_speed']
        avg_player_id1_speed = row[f'player_{id1}_average_player_speed']

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width = 350
        height = 230

        # Adjusting the position for right bottom corner
        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 300
        end_x = frame.shape[1] 
        end_y =  frame.shape[0]

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        text = f"     Player {chosen_player[0]}     Player {chosen_player[1]}"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_id0_shot_speed:.1f} km/h    {player_id1_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_id0_speed:.1f} km/h    {player_id1_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_id0_shot_speed:.1f} km/h    {avg_player_id1_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_id0_speed:.1f} km/h    {avg_player_id1_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_video_frames