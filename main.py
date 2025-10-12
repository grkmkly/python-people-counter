import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import supervision as sv
from ultralytics import YOLO

# --- SETTINGS ---
VIDEO_SOURCE = 0
MODEL_NAME = "yolov8n.pt"

def main():
    try:
        print("Starting sources")

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"ERROR: Don't open video source: ({VIDEO_SOURCE}) ")
            return

        ret, frame = cap.read()
        if not ret:
            print("ERROR: Not read first frame in video source .")
            return
        height, width, _ = frame.shape

        model = YOLO(MODEL_NAME)

        line_start = sv.Point(width // 2, 0)
        line_end = sv.Point(width // 2, height)
        line_zone = sv.LineZone(start=line_start, end=line_end)

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE)

        print("Starting counting...")

        while ret:
            # Track and Detect
            results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")[0]
            detections = sv.Detections.from_ultralytics(results)

            # If results id is not none, trigger
            if results.boxes.id is not None:
                detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
                line_zone.trigger(detections=detections)

            # Copy frame
            annotated_frame = frame.copy()
            
            # Annotate Frame for detection
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            # If Detection object is not none, annotate 
            if detections.tracker_id is not None:
                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )

            cv2.line(annotated_frame, line_start.as_xy_int_tuple(), line_end.as_xy_int_tuple(), (255, 255, 255), 2)
        
            total_count = line_zone.in_count + line_zone.out_count
            text = f"Toplam Gecis: {total_count}"
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Gecis Sayaci", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            ret, frame = cap.read()
    except Exception as e:
        print(f"Exception: {e}")
        
    finally:
        print("Closing program...")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()