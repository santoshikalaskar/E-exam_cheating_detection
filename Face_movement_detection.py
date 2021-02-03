import cv2
import numpy as np
import dlib

class FaceMovementDetection:

    def __init__(self):
        self.video_cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def StartWebCam(self):
        while True:
            _, self.frame = self.video_cap.read()
            self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.faces = self.detector(self.gray_frame)
            self.local_face_count = 0
            self.local_no_face_count = 0

            # No Face in Frame
            if not self.faces:
                self.local_no_face_count = self.local_no_face_count + 1
                cv2.putText(self.frame, "No Face Detected..! ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 90), 2)
            for face in self.faces:
                # count No of Faces
                self.local_face_count = self.local_face_count + 1
                if self.local_face_count > 1:
                    cv2.putText(self.frame, "Face Count: " + str(self.local_face_count), (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                self.landmarks = self.predictor(self.gray_frame, face)
                Left_x_point_3 = self.landmarks.part(2).x
                Left_y_point_3 = self.landmarks.part(2).y
                Center_x_point_31 = self.landmarks.part(30).x
                Center_y_point_31 = self.landmarks.part(30).y
                Right_x_point_15 = self.landmarks.part(14).x
                Right_y_point_15 = self.landmarks.part(14).y
                Main_diff_3_to_15 = int(np.sqrt((Left_x_point_3 - Right_x_point_15) ** 2 + (Left_y_point_3 - Right_y_point_15) ** 2))
                Left_side_diff = int(np.sqrt((Center_x_point_31 - Left_x_point_3) ** 2 + (Center_y_point_31 - Left_y_point_3) ** 2))
                Right_side_diff = int(np.sqrt((Center_x_point_31 - Right_x_point_15) ** 2 + (Center_y_point_31 - Right_y_point_15) ** 2))
                Left_side_percentage = int((Left_side_diff / Main_diff_3_to_15) * 100)
                Right_side_percentage = int((Right_side_diff / Main_diff_3_to_15) * 100)

                # Drawing Lines
                cv2.line(self.frame, (Left_x_point_3, Left_y_point_3), (Center_x_point_31, Center_y_point_31), (0, 255, 0), 3)
                cv2.line(self.frame, (Right_x_point_15, Right_y_point_15), (Center_x_point_31, Center_y_point_31), (0, 255, 0), 3)
                cv2.putText(self.frame, "Right side Distance" + str(Left_side_percentage), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 0), 2)
                cv2.putText(self.frame, "Left Side Distance" + str(Right_side_percentage), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 10), 2)

            cv2.imshow("Web_cam_on", self.frame)
            escap_key = cv2.waitKey(1)
            if escap_key == 27:
                break

if __name__ == "__main__":
    FaceMovementDetection_obj = FaceMovementDetection()
    FaceMovementDetection_obj.StartWebCam()