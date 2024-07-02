    player_heights = {
            chosen_players[0]: constants.PLAYER_1_HEIGHT_METERS,
            chosen_players[1]: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            if frame_num < len(ball_boxes):  # Ensure ball_boxes has data for this frame
                ball_box = ball_boxes[frame_num][1]
                ball_position = get_center_of_bbox(ball_box)
            else:
                ball_position = None  # Handle case where ball position is not available

            if not player_bbox:  # Skip frame if no players detected
                continue
            
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_key_point = (original_court_key_points[closest_key_point_index * 2],
                                    original_court_key_points[closest_key_point_index * 2 + 1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)

                bboxes_heights_in_pixels = []
                for i in range(frame_index_min, frame_index_max):
                    if player_id in player_boxes[i]:
                        bboxes_heights_in_pixels.append(get_height_of_bbox(player_boxes[i][player_id]))

                if bboxes_heights_in_pixels:
                    max_player_height_in_pixels = max(bboxes_heights_in_pixels)
                else:
                    max_player_height_in_pixels = constants.DEFAULT_HEIGHT_METERS  # Default height or handle as appropriate

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point,
                                                                            closest_key_point_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights.get(player_id, constants.DEFAULT_HEIGHT_METERS)
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id and ball_position is not None:
                    closest_key_point_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    closest_key_point = (original_court_key_points[closest_key_point_index * 2],
                                        original_court_key_points[closest_key_point_index * 2 + 1])

                    mini_court_ball_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point,
                                                                            closest_key_point_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights.get(player_id, constants.DEFAULT_HEIGHT_METERS)
                                                                            )
                    output_ball_boxes.append({1: mini_court_ball_position})
            
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes