import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

"""
    0: Nose 
    1: Left Eye 
    2: Right Eye 
    5: Left Shoulder 
    6: Right Shoulder
"""

def main():
    pose_model = YOLO("yolov8n-pose.pt") 

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to exit the program.")

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from the camera.")
            break

        frame_index += 1

        results = pose_model(frame, max_det=1)

        if results[0].boxes:
            annotated_frame = results[0].plot()

            person_width = results[0].boxes.xywh[0][2]
            print(f'Person width: {person_width}')

            left_eye = results[0].keypoints.data[0][1]
            right_eye = results[0].keypoints.data[0][2]
            distance = ((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)**0.5
            ratio = person_width/distance

            print(f'Left eye: {left_eye}')
            print(f'Right eye: {right_eye}')
            print(f'Distance between eyes: {distance}')

            print(f'Ratio: {ratio}')

            # save the frame_index and the ratio to a csv file
            with open('person_width_2_distance.csv', 'a') as f:
                f.write(f'{frame_index},{ratio:.3f}\n')

            visualize_ratio(frame_index, ratio)

            nose = results[0].keypoints.data[0][0]
            print(f'Nose: {nose}')
            left_shoulder = results[0].keypoints.data[0][5]
            right_shoulder = results[0].keypoints.data[0][6]

            print(f'Left Shoulder: {left_shoulder}')
            print(f'Right Shoulder: {right_shoulder}')

            shoulders_mid_point = ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2)
            print(f'Shoulders mid point: {shoulders_mid_point}')

            # distance between nose and shoulders through y axis
            nose_shoulder_distance = abs(nose[1] - shoulders_mid_point[1])
            print(f'Distance between nose and shoulders: {nose_shoulder_distance}')

            cv2.imshow("Pose Estimation", annotated_frame)

        else:
            cv2.imshow("Pose Estimation", frame)
            print("No person detected.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def visualize_ratio(frame_index, ratio):
    # visualize the ratio realtime as graph
    plt.plot(frame_index, ratio, 'ro-')
    plt.xlabel('Frame Index')
    plt.ylabel('Ratio')
    plt.title('Person Width to Distance Ratio')
    plt.grid()
    plt.pause(0.1)




if __name__ == "__main__":
    main()