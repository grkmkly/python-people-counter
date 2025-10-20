import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# Video and Mdel
VIDEO_SOURCE = 0
MODEL_NAME = "yolov8n.pt"
# Confidence Value
CONFIDENCE_THRESHOLD = 0.7 
# Count Window 
COUNT_WINDOW_WIDTH = 700
COUNT_WINDOW_HEIGHT = 700
# File Name
SAVE_FILE_NAME= "count.txt"

def load_count(file_name):
    try:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                counts = f.read().split(',')
                in_count = int(counts[0])
                out_count = int(counts[1])
                return in_count, out_count
    except Exception as e:
        print(f"ERROR: Can not read file{e}. Starting zero")
    
    print("Can not found file, Starting zero")
    return 0, 0

def save_count(file_name, in_c, out_c):
    try:
        with open(file_name, 'w') as f:
            f.write(f"{in_c},{out_c}")
    except Exception as e:
        print(f"Can not save to file {e}")

def main():
    try:
        print("Sources starting...")

        in_count, out_count = load_count(SAVE_FILE_NAME)

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"ERROR: Can not open video source: ({VIDEO_SOURCE})")
            return
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Can not read first frame")
            return
        # Get heigth and width
        height, width, _ = frame.shape
        # Model
        model = YOLO(MODEL_NAME)
        # Horizantal Line
        line_y_coordinate = int(height * 0.8) 
        line_start = (0, line_y_coordinate)
        line_end = (width, line_y_coordinate)
        # Annotors 
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.RED)

        tracker_positions = {}  

        print("Counting starting...")

        while ret:
            # Track model people
            results = model.track(
                frame, 
                persist=True, 
                classes=[0], 
                tracker="bytetrack.yaml",
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )[0]

            # YOLO result detect for supervision
            detections = sv.Detections.from_ultralytics(results)

            # If result not null, detection define id 
            if results.boxes.id is not None:
                detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
            # Frame Copy
            annotated_frame = frame.copy()
            # Draw box for detection result 
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            # Draw Line
            cv2.line(annotated_frame, line_start, line_end, (255, 255, 255), 2)

            # Current Id Set
            current_track_ids = set() 
            count_updated = False

            # If detection tracker is not null, process start 
            if detections.tracker_id is not None:
                labels = []
                for box, tracker_id in zip(detections.xyxy, detections.tracker_id):
                    
                    labels.append(f"ID: {tracker_id}")
                    current_track_ids.add(tracker_id)
                    # Body center 
                    body_center_y = int((box[1] + box[3]) / 2)
                    # Current position below or abowe
                    current_position = "above" if body_center_y < line_y_coordinate else "below"

                    # Save current position tracker
                    if tracker_id not in tracker_positions:
                        tracker_positions[tracker_id] = current_position
                    else:
                        prev_position = tracker_positions[tracker_id]
                        # If above in count increase
                        if prev_position == "above" and current_position == "below":
                            in_count += 1
                            count_updated = True
                        
                        # If below out count increase
                        elif prev_position == "below" and current_position == "above":
                            out_count += 1
                            count_updated = True
                        # Current Positioin
                        tracker_positions[tracker_id] = current_position
                # Draw ID box
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )

            if count_updated:
                save_count(SAVE_FILE_NAME, in_count, out_count)
            
            # Add remove list
            tracked_ids_to_remove = []
            for tracker_id in tracker_positions:
                if tracker_id not in current_track_ids:
                    tracked_ids_to_remove.append(tracker_id)
            # Delete tracker position
            for tracker_id in tracked_ids_to_remove:
                del tracker_positions[tracker_id]
            # Count Frame
            count_frame = np.zeros((COUNT_WINDOW_HEIGHT, COUNT_WINDOW_WIDTH, 3), dtype=np.uint8)
            # Text
            count_text = f"Giris: {in_count}" 
            
            # Font Styles
            font_scale = 5.0  
            font_thickness = 3 
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get font Styles
            text_size = cv2.getTextSize(count_text, font, font_scale, font_thickness)[0]
            
            # Centered In_count text 
            text_x = (COUNT_WINDOW_WIDTH - text_size[0]) // 2
            text_y = (COUNT_WINDOW_HEIGHT + text_size[1]) // 2
            cv2.putText(count_frame, count_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            cv2.imshow("Gecis Sayaci", annotated_frame)
            cv2.imshow("Giris Bilgisi", count_frame)

            # If press q, quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            ret, frame = cap.read()
            
    except Exception as e:
        print(f"Expected ERROR: {e}")
        
    finally:
        print("Sources release...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()