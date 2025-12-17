import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# --- SETTINGS ---
VIDEO_SOURCE = 0
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for detections
COUNT_WINDOW_WIDTH = 1280  # Width of the counting dashboard window
COUNT_WINDOW_HEIGHT = 720  # Height of the counting dashboard window
SAVE_FILE_NAME = "count.txt"
LOGO_PATH = "logo.jpg"  # Logo dosyası ekleme (varsa)

# --- COLORS ---
COLOR_BG = (40, 44, 52)        # Dark Grey 
COLOR_CARD_IN = (46, 139, 87)  # Sea Green (for In)
COLOR_CARD_OUT = (70, 70, 200) # Reddish (for Out)
COLOR_TEXT_MAIN = (255, 255, 255) # White
COLOR_TEXT_SEC = (200, 200, 200) # Light Grey

# Load count functions from txt file
def load_count(file_name):
    try:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                content = f.read().strip()
                if not content: return 0, 0
                counts = content.split(',')
                return int(counts[0]), int(counts[1])
    except Exception as e:
        print(f"ERROR: Can not read ({e}). Starting from zero.")
    return 0, 0

# Save count functions to txt file
def save_count(file_name, in_c, out_c):
    try:
        with open(file_name, 'w') as f:
            f.write(f"{in_c},{out_c}")
    except Exception as e:
        print(f"ERROR: Could not save ({e})")

# Add logo overlay function
def overlay_transparent(background, overlay, x, y, target_w=None):
    try:
        if target_w is not None:
            scale_ratio = target_w / overlay.shape[1]
            target_h = int(overlay.shape[0] * scale_ratio)
            overlay = cv2.resize(overlay, (target_w, target_h))

        h, w = overlay.shape[:2]

        if x + w > background.shape[1]: w = background.shape[1] - x
        if y + h > background.shape[0]: h = background.shape[0] - y
        if x < 0 or y < 0: return background 

        overlay_img = overlay[:h, :w]

        if overlay_img.shape[2] < 4:
            background[y:y+h, x:x+w] = overlay_img
            return background
        
        alpha_mask = overlay_img[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask

        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha_mask * overlay_img[:, :, c] + 
                                           alpha_inv * background[y:y+h, x:x+w, c])
        return background

    except Exception as e:
        print(f"Can not added logo {e}")
        return background

# Draw text centered at given coordinates
def draw_text_centered(img, text, font, scale, thickness, color, center_x, center_y):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = int(center_x - text_size[0] / 2)
    text_y = int(center_y + text_size[1] / 2)
    cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

# Create dashboard UI function
def create_dashboard_ui(width, height, in_count, out_count, logo_img=None):
    # Background
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = COLOR_BG

    # Header
    header_height = int(height * 0.15)
    cv2.rectangle(canvas, (0, 0), (width, header_height), (30, 30, 30), -1)
    
    # Logo
    if logo_img is not None:
        canvas = overlay_transparent(canvas, logo_img, x=30, y=15, target_w=100)

    draw_text_centered(canvas, "CANLI ZIYARETCI SISTEMI", cv2.FONT_HERSHEY_TRIPLEX, 
                       1.0, 2, COLOR_TEXT_MAIN, width // 2, header_height // 2)

    # in_count Card
    card_w = int(width * 0.50)
    card_h = int(height * 0.50)
    
    start_x = (width - card_w) // 2
    start_y = (height - card_h) // 2 + 20 

    # Background Card
    cv2.rectangle(canvas, (start_x, start_y), (start_x + card_w, start_y + card_h), COLOR_CARD_IN, -1)
    # Card Border
    cv2.rectangle(canvas, (start_x, start_y), (start_x + card_w, start_y + card_h), (255, 255, 255), 3)
    
    # Texts on Card
    draw_text_centered(canvas, "TOPLAM GIRIS", cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, 2, COLOR_TEXT_SEC, start_x + card_w // 2, start_y + 60)
    
    # in_count Number
    draw_text_centered(canvas, str(in_count), cv2.FONT_HERSHEY_DUPLEX, 
                       6.0, 8, COLOR_TEXT_MAIN, start_x + card_w // 2, start_y + card_h // 2 + 20)

    # out_count Text below in_count
    # footer_text = f"Cikis Yapan: {out_count} | Toplam Hareket: {in_count + out_count}"
    #draw_text_centered(canvas, footer_text, cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC, 0.6, 1, (150, 150, 150), width // 2, height - 30)

    return canvas

def main():
    try:
        print("Loading previous counts...")
        # Load previous counts
        in_count, out_count = load_count(SAVE_FILE_NAME)

        logo_img = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)

        if logo_img is None:
            print(f"INFO: {LOGO_PATH} not found, logo will not be displayed.")

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"ERROR: Video source could not be opened: ({VIDEO_SOURCE})")
            return

        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read the first frame")
            return

        # Load model
        model = YOLO(MODEL_NAME)
        
        # Annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.RED)
        tracker_positions = {}  

        print("Counting started...")

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret: break

            height, width, _ = frame.shape
            line_y_coordinate = int(height * 0.75) 
            line_start = (0, line_y_coordinate)
            line_end = (width, line_y_coordinate)

            # YOLO Tracking
            results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            if results.boxes.id is not None:
                detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)

            # --- ANNOTATIONS & COUNTING ---
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.line(annotated_frame, line_start, line_end, (0, 255, 255), 2) 

            current_track_ids = set()
            count_updated = False

            if detections.tracker_id is not None:
                labels = []
                for box, tracker_id in zip(detections.xyxy, detections.tracker_id):
                    labels.append(f"#{tracker_id}")
                    current_track_ids.add(tracker_id)
                    
                    # Calculate foot position
                    body_center_y = int(box[3]) 
                    
                    current_position = "above" if body_center_y < line_y_coordinate else "below"

                    if tracker_id not in tracker_positions:
                        tracker_positions[tracker_id] = current_position
                    else:
                        prev_position = tracker_positions[tracker_id]
                        
                        # IN 
                        if prev_position == "above" and current_position == "below":
                            in_count += 1
                            count_updated = True
                        # OUT 
                        elif prev_position == "below" and current_position == "above":
                            out_count += 1
                            count_updated = True
                            
                        tracker_positions[tracker_id] = current_position
                
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Save updated counts
            if count_updated:
                save_count(SAVE_FILE_NAME, in_count, out_count)

            # Cleanup (Memory Management)
            tracker_positions = {k: v for k, v in tracker_positions.items() if k in current_track_ids}

            # Dashboard Creation
            dashboard_frame = create_dashboard_ui(COUNT_WINDOW_WIDTH, COUNT_WINDOW_HEIGHT, in_count, out_count,logo_img=logo_img)

            # Show Windows
            cv2.imshow("Camera Tracking", annotated_frame)
            cv2.imshow("Control Panel", dashboard_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Releasing resources...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()