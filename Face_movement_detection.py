import cv2
import numpy as np
import dlib
import os
import face_recognition

class FaceMovementDetection:

    def __init__(self):
        self.video_cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.name = "santoshi kalaskar"
        self.training_image_path = os.path.join("Training_images", self.name)
        self.training_dataset_path = os.path.join("Training_dataset", self.name)
        self.local_no_face_count = 0
        self.global_no_face_counter = 0
        self.local_more_than_one_face_counter = 0
        self.global_more_than_one_face_counter = 0
        self.local_left_right_counter = 0
        self.global_left_right_counter = 0
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(self.fps))
        self.training_images, self.training_img_names = self.Create_Image_Name_list()

    def Create_Image_Name_list(self):
        self.training_images = []
        self.training_img_names = []
        self.myImageList = os.listdir(self.training_image_path)
        for img_list in self.myImageList:
            current_img = cv2.imread(f'{self.training_image_path}/{img_list}')
            self.training_images.append(current_img)
            self.training_img_names.append(os.path.splitext(img_list)[0])
        return self.training_images, self.training_img_names

    def StopExam(self,warning_msg):
        print(warning_msg)
        cv2.destroyAllWindows()
        self.video_cap.release()

    def StartWebCam(self):

        obama_image = face_recognition.load_image_file("Training_images/santoshi kalaskar/santoshi kalaskar.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        # Create arrays of known face encodings and their names
        known_face_encodings = [
            obama_face_encoding,
        ]
        known_face_names = [
            "Santoshi Kalaskar",
        ]
        # Initialize some variables
        face_locations = []
        face_names = []
        process_this_frame = True


        while True:
            _, self.frame = self.video_cap.read()

            # Creating object of Brightness class
            self.frame = cv2.convertScaleAbs(self.frame, alpha=1.5, beta=1)
            ''' ---------------------------------------------------------------------'''
            small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings1 = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings1:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Draw a box around the face
                cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(self.frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(self.frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ''' ---------------------------------------------------------------------'''
            self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.faces = self.detector(self.gray_frame)

            # if No Face Detected in Frame
            if not self.faces:
                self.local_no_face_count = self.local_no_face_count + 1
                if ( self.local_no_face_count % (5 * self.fps) ) == 0:
                    self.global_no_face_counter = self.global_no_face_counter + 1
                    if self.global_no_face_counter == 1:
                        cv2.putText(self.frame, "No Face Detected..! " + str(self.local_no_face_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 90), 2)
                        msg = "No one Detected...! stop exam ..!"
                        self.global_no_face_counter = 0
                        self.local_no_face_count = 0
                        self.StopExam(msg)
                        return 0

            self.local_face_count = 0
            for face in self.faces:
                # if count No of Faces more than one
                self.local_face_count = self.local_face_count + 1
                if self.local_face_count > 1:
                    self.local_more_than_one_face_counter = self.local_more_than_one_face_counter + 1
                    if (self.local_more_than_one_face_counter % (5 * self.fps)) == 0:
                        self.global_more_than_one_face_counter = self.global_more_than_one_face_counter + 1
                        if self.global_more_than_one_face_counter == 3:
                            cv2.putText(self.frame, "Face Count: " + str(self.local_face_count), (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            msg = "More than one Faces Detected...! stop exam ..!"
                            self.global_more_than_one_face_counter = 0
                            self.local_more_than_one_face_counter = 0
                            self.local_face_count  = 0
                            self.StopExam(msg)
                            return 0

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

                # if looking at left and right side
                if Left_side_percentage < 20 or Right_side_percentage < 20 :
                    self.local_left_right_counter = self.local_left_right_counter + 1
                    if (self.local_left_right_counter % (5 * self.fps)) == 0:
                        self.global_left_right_counter = self.global_left_right_counter + 1
                        if self.global_left_right_counter == 3:
                            cv2.putText(self.frame, "Right side Distance: " + str(Left_side_percentage), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 0), 2)
                            cv2.putText(self.frame, "Left Side Distance: " + str(Right_side_percentage), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 10), 2)
                            msg = "Face Movement Detected...! stop exam ..!"
                            self.global_left_right_counter = 0
                            self.local_left_right_counter = 0
                            self.StopExam(msg)
                            return 0

            cv2.imshow("Web_cam_on", self.frame)
            escap_key = cv2.waitKey(1)
            if escap_key == 27:
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    FaceMovementDetection_obj = FaceMovementDetection()
    FaceMovementDetection_obj.StartWebCam()