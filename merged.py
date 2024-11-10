import cv2
import time
import logging
import numpy as np
logging.getLogger("ultralytics").setLevel(logging.ERROR)


from utils import *

def main(save_video=False, duration=10, fps=6, camera=0):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    target_frame_count = duration * fps
    frame_duration = 1.0 / fps
    video_writer = initialize_video_writer(save_video=save_video)

    frame_count = 0
    while frame_count < target_frame_count:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_resized = cv2.resize(frame, (640, 480))
        
        # Process frames
        original_frame = frame_resized.copy()
        detection_frame = run_object_detection(frame_resized)
        depth_frame = run_depth_estimation(frame_resized, overlay_detection=True)
        seg_frame = run_semantic_segmentation(frame_resized, overlay_original=True)

        # Layout for 2x2 Grid
        top_row = np.hstack((original_frame, detection_frame))
        bottom_row = np.hstack((depth_frame, seg_frame))
        composite_frame = np.vstack((top_row, bottom_row))

        if save_video and video_writer:
            video_writer.write(composite_frame)

        # Display each window and composite
        cv2.imshow("Original Frame", original_frame)
        cv2.imshow("Object Detection", detection_frame)
        cv2.imshow("Depth Map with Detection", depth_frame)
        cv2.imshow("Original with Semantic Segmentation", seg_frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(save_video=False, duration=1000, fps=7, camera=1)