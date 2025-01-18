import cv2
from ultralytics import YOLO

def main():
    # Load the YOLO model pre-trained for pose estimation
    pose_model = YOLO("yolov8n-pose.pt") 
    # face_model = YOLO("yolov8n-oiv7.pt")

    # Open a connection to the camera (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to exit the program.")

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from the camera.")
            break

        # Perform pose estimation
        results = pose_model(frame, max_det=1)
        # face_results = face_model(frame, max_det=1)

        # Annotate the frame with pose estimation results
        annotated_frame = results[0].plot()

        # add the head detection to the frame as well
        # x1, y1, x2, y2 = map(int, face_results[0].boxes[0].xyxy[0])
        # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # print the bbox
        # print(f'results[0]: {results[0].boxes.xywh}')

        person_width = results[0].boxes.xywh[0][2]
        print(f'Person width: {person_width}')

        """
        0: Nose 
        1: Left Eye 
        2: Right Eye 
        5: Left Shoulder 
        6: Right Shoulder
        """

        # distance between eyes
        left_eye = results[0].keypoints.data[0][1]
        right_eye = results[0].keypoints.data[0][2]
        distance = ((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)**0.5

        print(f'Left eye: {left_eye}')
        print(f'Right eye: {right_eye}')
        print(f'Distance between eyes: {distance}')

        print(f'Ratio: {person_width/distance}')

        nose = results[0].keypoints.data[0][0]
        print(f'Nose: {nose}')
        left_shoulder = results[0].keypoints.data[0][5]
        right_shoulder = results[0].keypoints.data[0][6]

        print(f'Left Shoulder: {left_shoulder}')
        print(f'Right Shoulder: {right_shoulder}')

        # mid point between shoulders
        shoulders_mid_point = ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2)
        print(f'Shoulders mid point: {shoulders_mid_point}')

        # distance between nose and shoulders through y axis
        nose_shoulder_distance = abs(nose[1] - shoulders_mid_point[1])
        print(f'Distance between nose and shoulders: {nose_shoulder_distance}')


        # Display the annotated frame
        cv2.imshow("Pose Estimation", annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
