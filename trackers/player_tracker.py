from ultralytics import YOLO
import cv2
import pickle

def get_centre_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    centre_x = int((x1+x2)/2)
    centre_y = int((y1+y2)/2)
    return (centre_x,centre_y)
def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        keypoints = [court_keypoints[26],court_keypoints[27],court_keypoints[24],court_keypoints[25],court_keypoints[4],court_keypoints[5],court_keypoints[10],court_keypoints[11],court_keypoints[14],court_keypoints[15],court_keypoints[6],court_keypoints[7]]
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections, chosen_player


    def choose_players(self, court_keypoints, player_dict):
        if not player_dict:  # Check if player_dict is empty
            return []

        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_centre_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Choose the first 2 tracks if distances is not empty
        chosen_players = [distances[0][0], distances[1][0]] 
        return chosen_players



    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        last_valid_track_id = None

        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                last_valid_track_id = track_id
            else:
                if last_valid_track_id is None:
                    print("Warning: box.id is None and no previous valid track_id available")
                    continue
                track_id = last_valid_track_id
                print(f"Warning: box.id is None, using last valid track_id: {track_id}")

            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict



    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
