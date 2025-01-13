
import cv2
import numpy as np

class KalmanFilterLaneDetection:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32) * 0.03
        
        self.prev_left_lines = []
        self.prev_right_lines = []
        self.smoothing_factor = 0.9  
        self.kalman_gain = 0.6  

    def smooth_lines(self, current_lines, prev_lines):
        current_lines = np.array(current_lines)
        prev_lines = np.array(prev_lines)

        if len(prev_lines) > 0 and len(current_lines) == len(prev_lines):
            smoothed_lines = []
            for i in range(len(current_lines)):
                smoothed_line = current_lines[i] * self.smoothing_factor + prev_lines[i] * (1 - self.smoothing_factor)
                smoothed_lines.append(smoothed_line)
            return smoothed_lines
        else:
            return current_lines.tolist()  

    def kalman_update(self, detected_lines, prev_lines):
        if len(prev_lines) > 0:
            predicted_lines = self.kalman.predict()
            measurement = np.array(detected_lines).reshape(-1, 1)
            self.kalman.correct(measurement)
            updated_state = self.kalman.statePost.reshape(-1)

            return updated_state
        else:
            return detected_lines
        

    def calculate_curvature(self, line):
        x1, y1, x2, y2 = line[0]
        # Fit a second-order polynomial (y = ax^2 + bx + c) to the line
        fit = np.polyfit([y1, y2], [x1, x2], 2)
        #curvature = (1 + (2Ay + B)^2)^1.5 / |2A|
        curvature = (1 + (2 * fit[0] * y1 + fit[1])**2)**1.5 / np.abs(2 * fit[0])
        return curvature

    def detect_curved_lines_in_lane(self, image, edges, lane_roi):
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
                curvature = self.calculate_curvature(line)


                if abs(slope) > 0.2 and curvature > 100: #0.2, 100
                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if line_length > 50:  #50
                        filtered_lines.append(line)
            
            return filtered_lines
        else:
            return None

    def preprocess_frame(self, frame):
        processed_frame = cv2.resize(frame, (1164, 874))
        return processed_frame

    def create_lane_roi(self, image):
        vertices = np.array([[(150, 874), (1064, 874), (682, 400), (482, 400)]], dtype=np.int32)
        mask = np.zeros_like(image)
      
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        
        return mask

    def calculate_pitch_and_yaw(self, image, averaged_lines):
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
            return [0,0]

        horizon_line = [image.shape[1] // 2, 0, image.shape[1] // 2, image.shape[0]]

        pitch_angle = np.arctan2(vanishing_point[1] - horizon_line[1], vanishing_point[0] - horizon_line[0])
        ground_vanishing_point = [vanishing_point[0], vanishing_point[1], 0]

        horizon_vanishing_point = [vanishing_point[0], image.shape[0] // 2, vanishing_point[1]]

        yaw_angle = np.arctan2(horizon_vanishing_point[2] - image.shape[1] // 2,
                               horizon_vanishing_point[0] - image.shape[0] // 2)
        
        if abs(pitch_angle) < 0.1 or abs(yaw_angle) < 0.1:
            return 0, 0


        return pitch_angle, yaw_angle
    
    def estimate_speed(self, prev_frame, current_frame):
        # Compute optical flow between previous and current frames
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract flow vectors and calculate their magnitudes
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        flow_magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)

        # Compute the average speed
        average_speed = np.mean(flow_magnitude)

        return average_speed

    def enforce_parallelism(self, left_line, right_line):
        if len(left_line) != 4 or len(right_line) != 4:
            
            return left_line, right_line

        # Calculate vanishing point (intersection of left and right lines)
        try:
            vanishing_point = np.linalg.solve(
                [[left_line[1] - left_line[3], left_line[0] - left_line[2]],
                 [right_line[1] - right_line[3], right_line[0] - right_line[2]]],
                [left_line[0] * (left_line[1] - left_line[3]) - left_line[1] * (left_line[0] - left_line[2]),
                 right_line[0] * (right_line[1] - right_line[3]) - right_line[1] * (right_line[0] - right_line[2])]
            )
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Unable to calculate vanishing point.")
            return left_line, right_line

        # Calculate horizon line (middle of the image)
        horizon_line = [self.image_shape[1] // 2, 0, self.image_shape[1] // 2, self.image_shape[0]]

        # Calculate pitch angle
        pitch_angle = np.arctan2(vanishing_point[1] - horizon_line[1], vanishing_point[0] - horizon_line[0])

        # Calculate left line slope
        slope_left = np.arctan2(left_line[3] - left_line[1], left_line[2] - left_line[0])

        # Calculate right line slope
        slope_right = np.arctan2(right_line[3] - right_line[1], right_line[2] - right_line[0])

        # Calculate average slope
        avg_slope = (slope_left + slope_right) / 2

        # Update left line coordinates
        left_line[2] = left_line[0] + (left_line[3] - left_line[1]) / np.tan(avg_slope)
        
        return left_line, right_line
    

    def interpolate_missing_segments(self, points):
        interpolated_points = []
        DESIRED_SEGMENT_LENGTH = 2.0 #2.0-2.5 m
        MAX_DISTANCE_THRESHOLD =  2.0 #2-3 m
        for i in range(len(points) - 1):
            # Calculate the distance between consecutive points
            dist = np.sqrt((points[i+1][0] - points[i][0]) ** 2 + (points[i+1][1] - points[i][1]) ** 2)
            if dist < MAX_DISTANCE_THRESHOLD:  # Define a threshold for maximum distance between points
                # Interpolate points between the consecutive points
                num_interpolated_points = int(dist / DESIRED_SEGMENT_LENGTH)
                x_interp = np.linspace(points[i][0], points[i+1][0], num=num_interpolated_points+2)[1:-1]
                y_interp = np.linspace(points[i][1], points[i+1][1], num=num_interpolated_points+2)[1:-1]
                interpolated_points.extend(zip(x_interp, y_interp))
        return interpolated_points




    def run_lane_detection(self, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        #previous pos matters
        prev_left_fit = None
        prev_right_fit = None

        prev_frame = None
        estimated_speed = 0

        angles_array = []
        threshold = 150 

     
        left_fit = None
        right_fit = None


        while True:
          
            ret, frame = cap.read()

            if not ret:
                break

            processed_frame = self.preprocess_frame(frame)
            lane_roi = self.create_lane_roi(processed_frame)
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray, 50, 150)

            curved_lines = self.detect_curved_lines_in_lane(processed_frame, edges, lane_roi)

            if prev_frame is not None:
                estimated_speed = self.estimate_speed(prev_frame, processed_frame)

            prev_frame = processed_frame

            #Draw steady lane lines
            if curved_lines is not None:

                smoothed_lines = self.smooth_lines(curved_lines, self.prev_left_lines if len(self.prev_left_lines) > 0 else [])
                self.prev_left_lines = smoothed_lines
                smoothed_lines = self.smooth_lines(curved_lines, self.prev_right_lines if len(self.prev_right_lines) > 0 else [])
                self.prev_right_lines = smoothed_lines

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

                #Fit a polynomial to the points if points are available for both left and right lanes
                if pts_left and pts_right:
                    
                    pts_left = np.array(pts_left)
                    pts_right = np.array(pts_right)
                    left_fit_new = np.polyfit(pts_left[:, 1], pts_left[:, 0], 1)
                    right_fit_new = np.polyfit(pts_right[:, 1], pts_right[:, 0], 1)

                
                    left_fit_new, right_fit_new = self.enforce_parallelism(left_fit_new, right_fit_new)

                    print("left_fit_new:", left_fit_new)
                    print("right_fit_new:", right_fit_new)

   
                    if left_fit_new is not None and right_fit_new is not None:
                        prev_left_fit = left_fit_new
                        prev_right_fit = right_fit_new
                        left_fit = left_fit_new
                        right_fit = right_fit_new
                    else:
                        continue 
                else:
                    continue
            
                        
            print("left_fit:", left_fit)
            print("right_fit:", right_fit)

            ploty = np.linspace(0, processed_frame.shape[0]-1, processed_frame.shape[0])
            left_fitx = left_fit[0]*ploty + left_fit[1]
            right_fitx = right_fit[0]*ploty + right_fit[1]

            #test draw lane line
            for i in range(len(ploty)-1):
                cv2.line(processed_frame, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (0, 255, 0), 2)
                cv2.line(processed_frame, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 0), 2)

     
            averaged_lines = [(left_fitx[0], ploty[0], left_fitx[-1], ploty[-1]), (right_fitx[0], ploty[0], right_fitx[-1], ploty[-1])]
            pitch, yaw = self.calculate_pitch_and_yaw(processed_frame, averaged_lines)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            if pitch == 0 and yaw == 0:
                angles_array.append([0, 0])
            else:
                angles_array.append([abs(pitch_rad), abs(yaw_rad)])

            cv2.imshow('Curved Lines Detection', processed_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        angles_array = np.array(angles_array)
        print("Output Data (Pitch, Yaw):")
        print(angles_array)

        np.savetxt('9.txt', angles_array, fmt='%.6f', delimiter=' ', header='', comments='')

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    angles_array = []
    lane_detection = KalmanFilterLaneDetection()
    lane_detection.run_lane_detection('labeled/0.hevc')