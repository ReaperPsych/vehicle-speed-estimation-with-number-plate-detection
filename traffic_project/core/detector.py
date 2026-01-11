import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict
import math
import time

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # --- 1. DYNAMIC MODEL LOADING ---
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        
        print("Initializing OCR model...")
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
        self.target_classes = [2, 3, 5, 7] 
        self.conf_threshold = 0.5
        
        self.track_history = defaultdict(lambda: []) 
        # Add 'in_zone' flag to metadata to track vehicles that actually entered the measurement area
        self.vehicle_metadata = defaultdict(lambda: {'plate': 'Scanning...', 'speed_history': [], 'total_frames': 0, 'speed': 0, 'in_zone': False}) 
        
        self.FPS = 30  
        self.M = None 
        self.SOURCE_POINTS = None 
        self.FRAME_SKIP_SPEED_CHECK = 5 
        self.FRAME_SKIP_PLATE_READ = 30 
        
        self.MIN_PIXEL_HEIGHT = 30 
        self.SPEED_SMOOTHING_WINDOW = 12 

    def get_homography_matrix(self, source_points, real_width, real_height):
        """
        Calculates the homography matrix M dynamically.
        Now accepts real_width and real_height from user input.
        Respects Order: TL -> TR -> BR -> BL
        """
        try:
            # 1. Source Points (Pixels from User Click: TL, TR, BR, BL)
            SOURCE_POINTS = np.float32(source_points) 
            
            # 2. Destination Points (Meters) - Matched to TL, TR, BR, BL
            DEST_POINTS = np.float32([
                [0, 0],                  # Top-Left
                [real_width, 0],         # Top-Right
                [real_width, real_height], # Bottom-Right (Matched to 3rd click)
                [0, real_height]         # Bottom-Left (Matched to 4th click)
            ])

            M = cv2.getPerspectiveTransform(SOURCE_POINTS, DEST_POINTS)
            self.M = M
            self.SOURCE_POINTS = SOURCE_POINTS
            print(f"Homography Matrix calculated for area: {real_width}m x {real_height}m")
        except Exception as e:
            print(f"Error calculating Homography Matrix: {e}")
            self.M = None

    def read_plate(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        margin = 10
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(image.shape[1], x2 + margin)
        crop_y2 = min(image.shape[0], y2 + margin)
        
        vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if vehicle_crop.size == 0: return None
            
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ', detail=0)
        
        detected_text = ""
        if results:
            detected_text = max(results, key=len)
            
        cleaned_text = ''.join(filter(str.isalnum, detected_text)).upper()
        return cleaned_text if len(cleaned_text) >= 4 else "Unknown"

    def estimate_speed(self, track_id, current_pos_metric):
        history = self.track_history[track_id]
        if len(history) < self.FRAME_SKIP_SPEED_CHECK: return 0
            
        prev_pos_metric = history[-self.FRAME_SKIP_SPEED_CHECK]
        real_dist = math.sqrt(
            (current_pos_metric[0] - prev_pos_metric[0])**2 + 
            (current_pos_metric[1] - prev_pos_metric[1])**2
        )
        time_secs = self.FRAME_SKIP_SPEED_CHECK / self.FPS
        speed_kmh = (real_dist / time_secs) * 3.6
        
        self.vehicle_metadata[track_id]['speed_history'].append(speed_kmh)
        
        speeds = self.vehicle_metadata[track_id]['speed_history']
        window = self.SPEED_SMOOTHING_WINDOW
        smooth_speed = np.mean(speeds[-window:]) if len(speeds) > 0 else 0
        
        return int(smooth_speed)

    def process_video(self, source_path, output_path, source_points_list, real_width, real_height):
        """
        Main processing loop. Now accepts real dimensions.
        """
        start_time = time.time()
        
        cap = cv2.VideoCapture(source_path)
        self.FPS = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.FPS, (width, height))
        
        # Initialize Homography with user inputs
        self.get_homography_matrix(source_points_list, real_width, real_height)
        
        if self.M is None:
            raise ValueError("Calibration failed.")
            
        # Start Y is the minimum Y coordinate of the user's 4 points (the "back" line)
        calibration_start_y = min(p[1] for p in self.SOURCE_POINTS)
            
        frame_count = 0
        processing_start_time = time.time()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_count += 1
            
            # --- PROGRESS LOGGING ---
            if frame_count % 30 == 0:
                elapsed_since_start = time.time() - processing_start_time
                current_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
                video_timestamp = frame_count / self.FPS
                print(f"Processing Frame {frame_count} | Video Time: {video_timestamp:.1f}s | Processing Speed: {current_fps:.1f} FPS")

            results = self.model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().int().numpy()
                track_ids = results[0].boxes.id.cpu().int().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, cls_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                    if cls_id not in self.target_classes or conf < self.conf_threshold:
                        continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    if (y2 - y1) < self.MIN_PIXEL_HEIGHT: continue

                    cx, cy = (x1 + x2) // 2, y2 
                    
                    point_pixel = np.array([[[cx, cy]]], dtype=np.float32)
                    transformed_point = cv2.perspectiveTransform(point_pixel, self.M)[0][0]
                    mx, my = transformed_point[0], transformed_point[1]
                    
                    self.track_history[track_id].append([mx, my])
                    if len(self.track_history[track_id]) > self.FRAME_SKIP_SPEED_CHECK * 4:
                        self.track_history[track_id].pop(0)

                    self.vehicle_metadata[track_id]['total_frames'] += 1
                    
                    current_speed = self.vehicle_metadata[track_id]['speed']
                    
                    # --- ZONE ENTRY CHECK ---
                    if cy > calibration_start_y:
                        self.vehicle_metadata[track_id]['in_zone'] = True
                        
                        new_speed = self.estimate_speed(track_id, (mx, my))
                        if new_speed > 0:
                            current_speed = new_speed
                            self.vehicle_metadata[track_id]['speed'] = current_speed

                    # OCR Logic
                    current_plate = self.vehicle_metadata[track_id]['plate']
                    if (frame_count % self.FRAME_SKIP_PLATE_READ == 0 or current_plate == 'Scanning...'):
                        if (y2 - y1) > 100: 
                            plate_text = self.read_plate(frame, box)
                            if plate_text and plate_text != "Unknown":
                                self.vehicle_metadata[track_id]['plate'] = plate_text
                                current_plate = plate_text
                    
                    # Visualization
                    color = (0, 255, 0) 
                    if current_speed > 60: color = (0, 0, 255) 
                    elif current_speed > 40: color = (0, 165, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_speed = f"{current_speed} km/h" if current_speed > 0 else "Tracking..."
                    label_plate = f"{current_plate}"
                    
                    t_size_speed = cv2.getTextSize(label_speed, 0, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size_speed[0], y1), color, -1)
                    cv2.putText(frame, label_speed, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    t_size_plate = cv2.getTextSize(label_plate, 0, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y2), (x1 + t_size_plate[0] + 5, y2 + 20), color, -1)
                    cv2.putText(frame, label_plate, (x1 + 3, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw Calibration Zone
            if self.SOURCE_POINTS is not None:
                pts = self.SOURCE_POINTS.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
                cv2.putText(frame, f"ZONE: {real_width}m x {real_height}m", (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            out.write(frame)
            
        cap.release()
        out.release()
        end_time = time.time()

        # --- Stats Calculation ---
        vehicles_in_zone = len([v for v in self.vehicle_metadata.values() if v['in_zone']])
        
        final_speeds = [data['speed_history'][-1] for data in self.vehicle_metadata.values() if data['speed_history']]
        avg_speed = int(np.mean(final_speeds)) if final_speeds else 0
        max_speed = int(np.max(final_speeds)) if final_speeds else 0

        video_duration_secs = frame_count / self.FPS
        vehicles_per_minute = (vehicles_in_zone / video_duration_secs) * 60 if video_duration_secs > 0 else 0
        
        traffic_flow_level = "Light"
        traffic_flow_color = "green"
        if vehicles_per_minute > 20:
            traffic_flow_level = "Medium"
            traffic_flow_color = "orange"
        if vehicles_per_minute > 40:
            traffic_flow_level = "Heavy"
            traffic_flow_color = "red"
            
        return {
            'total_vehicles': vehicles_in_zone, 
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'traffic_flow_level': traffic_flow_level,
            'traffic_flow_color': traffic_flow_color,
            'processing_time': round(end_time - start_time, 2),
        }