#Without KalmanFilter


import cv2
import numpy as np

def detect_curved_lines_in_lane(image, edges, lane_roi):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lane_roi_gray = cv2.cvtColor(lane_roi, cv2.COLOR_BGR2GRAY)
    _, lane_mask = cv2.threshold(lane_roi_gray, 1, 255, cv2.THRESH_BINARY)
    lane_mask = cv2.resize(lane_mask, (edges.shape[1], edges.shape[0]))
    masked_edges = cv2.bitwise_and(edges, edges, mask=lane_mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=20)
    
    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  
            
            #filter out horizontal lines, doesn't make sense to have horizontal lines in lane detection
            if abs(slope) > 0.2:  #good enough
                # Filter out short lines
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length > 50:  #eh
                    filtered_lines.append(line)
        
        return filtered_lines
    else:
        return None


def preprocess_frame(frame):
    processed_frame = cv2.resize(frame, (1164, 874))
    return processed_frame

def create_lane_roi(image):
    vertices = np.array([[(150, 874), (1064, 874), (682, 400), (482, 400)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    
    return mask


def calculate_pitch_and_yaw(image, averaged_lines):
    left_line = averaged_lines[0]
    right_line = averaged_lines[1]

    x1_left, y1_left, x2_left, y2_left = left_line
    x1_right, y1_right, x2_right, y2_right = right_line

    try:
        vanishing_point = np.linalg.solve(
            [[left_line[1] - left_line[3], left_line[0] - left_line[2]],
             [right_line[1] - right_line[3], right_line[0] - right_line[2]]],
            [left_line[0] * (left_line[1] - left_line[3]) - left_line[1] * (left_line[0] - left_line[2]),
             right_line[0] * (right_line[1] - right_line[3]) - right_line[1] * (right_line[0] - right_line[2])]
        )
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Unable to calculate vanishing point.")
        return None


    horizon_line = [image.shape[1] // 2, 0, image.shape[1] // 2, image.shape[0]]
    pitch_angle = np.arctan2(vanishing_point[1] - horizon_line[1], vanishing_point[0] - horizon_line[0])

    # Project vanishing point onto the ground plane, camera height is z
    ground_vanishing_point = [vanishing_point[0], vanishing_point[1], 0]
    # Project vanishing point onto the horizon line
    horizon_vanishing_point = [vanishing_point[0], image.shape[0] // 2, vanishing_point[1]]
    # Calculate yaw angle
    yaw_angle = np.arctan2(horizon_vanishing_point[2] - image.shape[1] // 2,
                           horizon_vanishing_point[0] - image.shape[0] // 2)
    
    

    return pitch_angle, yaw_angle


video_path = '/unlabeled/9.hevc'


angles_array = []


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# previous lane positions
prev_left_fit = None
prev_right_fit = None



while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = preprocess_frame(frame)
    lane_roi = create_lane_roi(processed_frame)

    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    curved_lines = detect_curved_lines_in_lane(processed_frame, edges, lane_roi)

    if curved_lines is not None:
        pts_left = []
        pts_right = []
        for line in curved_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < 0:
                pts_left.append((x1, y1))
                pts_left.append((x2, y2))
            else:
                pts_right.append((x1, y1))
                pts_right.append((x2, y2))
        
        if pts_left and pts_right:
            pts_left = np.array(pts_left)
            pts_right = np.array(pts_right)
            
            left_fit = np.polyfit(pts_left[:, 1], pts_left[:, 0], 1)
            right_fit = np.polyfit(pts_right[:, 1], pts_right[:, 0], 1)
            
            prev_left_fit = left_fit
            prev_right_fit = right_fit
        elif prev_left_fit is not None and prev_right_fit is not None:
            left_fit = prev_left_fit
            right_fit = prev_right_fit
        else:
            continue
        
        ploty = np.linspace(0, processed_frame.shape[0]-1, processed_frame.shape[0])
        left_fitx = left_fit[0]*ploty + left_fit[1]
        right_fitx = right_fit[0]*ploty + right_fit[1]
        
        for i in range(len(ploty)-1):
            cv2.line(processed_frame, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (0, 255, 0), 2)
            cv2.line(processed_frame, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 0), 2)
        
        averaged_lines = [(left_fitx[0], ploty[0], left_fitx[-1], ploty[-1]), (right_fitx[0], ploty[0], right_fitx[-1], ploty[-1])]
        pitch, yaw = calculate_pitch_and_yaw(processed_frame, averaged_lines)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        angles_array.append([abs(pitch_rad), abs(yaw_rad)])
    else:
        angles_array.append([0,0])

    cv2.imshow('Curved Lines Detection', processed_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

angles_array = np.array(angles_array)
print("Output Data (Pitch, Yaw):")
print(angles_array)

np.savetxt('9.txt', angles_array, fmt='%.6f', delimiter=' ', header='', comments='')

cap.release()
cv2.destroyAllWindows()