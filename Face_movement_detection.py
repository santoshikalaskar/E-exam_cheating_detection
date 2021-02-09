
try:
    import cv2
    import numpy as np
    import dlib
    import os
    import face_recognition
    import logger_hander
    logger = logger_hander.set_logger()
    logger.info("Import all library successfully..!")

except ImportError as error:
    logger.error(error.__class__.__name__ + ": " + error.message)
except Exception as exception:
    logger.error(exception.__class__.__name__ + ": " + exception.message)

class Face_Movement_Detection:

    def __init__(self):
        self.Initialize_Counter_Veriable()
        self.Initialize_Dlib_Lib()
        self.Get_Web_Cam()

    def Initialize_Counter_Veriable(self):
        try:
            self.internal_no_face_counter = 0
            self.outer_no_face_counter = 0
            self.internal_more_than_one_face_counter = 0
            self.outer_more_than_one_face_counter = 0
            self.internal_left_right_counter = 0
            self.outer_left_right_counter = 0
            self.internal_unknown_face_counter = 0
            self.outer_unknown_face_counter = 0
        except Exception as exception:
            logger.error(exception.__class__.__name__ + ": " + exception.message)


    def Initialize_Dlib_Lib(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            logger.info(" Dlib Detector and shape_predictor_68_face_landmarks shape predictor set...! ")
        except AttributeError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except ModuleNotFoundError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except Exception as exception:
            logger.error(exception.__class__.__name__ + ": " + exception.message)

    def Get_Web_Cam(self):
        try:
            self.video_cap = cv2.VideoCapture(0)
            logger.info(" Got Video Camera...! ")
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            logger.info(" Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(self.fps))
        except ModuleNotFoundError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except SystemError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except InterruptedError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except Exception as exception:
            logger.error(exception.__class__.__name__ + ": " + exception.message)

    def Get_Training_Data(self):
        try:
            name = "santoshi kalaskar"
            training_image_path = os.path.join("Training_images", name)
            # self.training_dataset_path = os.path.join("Training_dataset", self.name)
            self.training_images = []
            self.training_img_names = []
            self.training_face_encoding = []
            myImageList = os.listdir(training_image_path)
            for img_list in myImageList:
                current_img = cv2.imread(f'{training_image_path}/{img_list}')
                current_img_path = os.path.join(training_image_path, img_list)
                current_image = face_recognition.load_image_file(current_img_path)
                self.training_images.append(current_img)
                self.training_img_names.append(os.path.splitext(img_list)[0])
                self.training_face_encoding.append(face_recognition.face_encodings(current_image)[0])
                logger.info(" Training Images and Training Face encodings done ...! ")
            # return self.training_images, self.training_img_names, self.training_face_encoding
        except FileExistsError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except FileNotFoundError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except Exception as exception:
            logger.error(exception.__class__.__name__ + ": " + exception.message)

    def Stop_Exam(self,warning_msg):
        try:
            logger.info(warning_msg)
            cv2.destroyAllWindows()
            self.video_cap.release()
        except ModuleNotFoundError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except SystemError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except InterruptedError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except Exception as exception:
            logger.error(exception.__class__.__name__ + ": " + exception.message)

    def Start_Web_Cam(self):
        try:
            while True:
                _, self.frame = self.video_cap.read()

                ''' ---------------------------------------------------------------------
                                    Frame Brightness Logic
                ---------------------------------------------------------------------'''
                # Creating object of Brightness class
                # self.frame = cv2.convertScaleAbs(self.frame, alpha=1.0, beta=1)


                ''' ---------------------------------------------------------------------
                                    No Face Detected in Frame Logic
                ---------------------------------------------------------------------'''
                # convert BGR frame to gray scale
                gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # fetch faces from frame
                self.faces = self.detector(gray_frame)
                if not self.faces:
                    self.internal_no_face_counter += 1
                    if ( self.internal_no_face_counter % (5 * self.fps) ) == 0:
                        self.outer_no_face_counter += 1
                        if self.outer_no_face_counter == 3:
                            cv2.putText(self.frame, "No Face Detected..! " + str(self.internal_no_face_counter), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 90), 2)
                            msg = " No one Detected...! stop exam ..!"
                            self.outer_no_face_counter = 0
                            self.internal_no_face_counter = 0
                            self.Stop_Exam(msg)
                            break

                if self.faces:

                    ''' ---------------------------------------------------------------------
                                        Face Identification Logic
                    ---------------------------------------------------------------------'''
                    # Resize Frame in 1/4 size for better accuracy.
                    small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    # find face locations
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    # find face encodings
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    name = "Unknown"
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(self.training_face_encoding, face_encoding)
                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(self.training_face_encoding, face_encoding)
                        if round(min(face_distances), 2) < 0.58:
                            best_match_index = np.argmin(face_distances)
                            # get name of that matched face
                            if matches[best_match_index]:
                                name = self.training_img_names[best_match_index]
                        face_names.append(name)
                    if name == "Unknown":
                        self.internal_unknown_face_counter += 1
                        if (self.internal_unknown_face_counter % (5 * self.fps)) == 0:
                            self.outer_unknown_face_counter += 1
                            if self.outer_unknown_face_counter == 3:
                                cv2.putText(self.frame, "Unknown Face Detected..! " + str(self.internal_no_face_counter),
                                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 90), 2)
                                msg = " Unknown one Detected...! stop exam ..!"
                                self.outer_unknown_face_counter = 0
                                self.internal_unknown_face_counter = 0
                                self.Stop_Exam(msg)
                                break

                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        # Draw a box around the face
                        cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Draw a box below face box
                        cv2.rectangle(self.frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        # Draw a label with a name below the face
                        cv2.putText(self.frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),1)

                    ''' ---------------------------------------------------------------------
                                        More than One Face Detected in Frame Logic
                     ---------------------------------------------------------------------'''
                    self.internal_face_count = 0
                    # check for more than one face detected
                    for face in self.faces:
                        self.internal_face_count += 1
                        if self.internal_face_count > 1:
                            self.internal_more_than_one_face_counter += 1
                            if (self.internal_more_than_one_face_counter % (5 * self.fps)) == 0:
                                self.outer_more_than_one_face_counter += 1
                                if self.outer_more_than_one_face_counter == 3:
                                    cv2.putText(self.frame, "Face Count: " + str(self.internal_face_count), (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    msg = "More than one Faces Detected...! stop exam ..!"
                                    self.outer_more_than_one_face_counter = 0
                                    self.internal_more_than_one_face_counter = 0
                                    self.internal_face_count  = 0
                                    self.Stop_Exam(msg)
                                    break

                        ''' ---------------------------------------------------------------------
                                looking at Left and Right side Detected in Frame Logic
                        ---------------------------------------------------------------------'''
                        # Estimate the location of 68 co-ordinates to map facial points
                        self.landmarks = self.predictor(gray_frame, face)
                        left_x_point_3 = self.landmarks.part(2).x
                        left_y_point_3 = self.landmarks.part(2).y
                        center_x_point_31 = self.landmarks.part(30).x
                        center_y_point_31 = self.landmarks.part(30).y
                        right_x_point_15 = self.landmarks.part(14).x
                        right_y_point_15 = self.landmarks.part(14).y
                        # find Total distance
                        main_diff_3_to_15 = int(np.sqrt((left_x_point_3 - right_x_point_15) ** 2 + (left_y_point_3 - right_y_point_15) ** 2))
                        # find Left side distance
                        left_side_diff = int(np.sqrt((center_x_point_31 - left_x_point_3) ** 2 + (center_y_point_31 - left_y_point_3) ** 2))
                        # find Right side distance
                        right_side_diff = int(np.sqrt((center_x_point_31 - right_x_point_15) ** 2 + (center_y_point_31 - right_y_point_15) ** 2))
                        # find Left side distance percentage(%)
                        left_side_percentage = int((left_side_diff / main_diff_3_to_15) * 100)
                        # find Right side distance percentage(%)
                        right_side_percentage = int((right_side_diff / main_diff_3_to_15) * 100)
                        # set threshold limit of left and right side %
                        if left_side_percentage < 20 or right_side_percentage < 20 :
                            self.internal_left_right_counter += 1
                            if (self.internal_left_right_counter % (5 * self.fps)) == 0:
                                self.outer_left_right_counter += 1
                                if self.outer_left_right_counter == 3:
                                    cv2.putText(self.frame, "Right side Distance: " + str(left_side_percentage), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 210, 0), 2)
                                    cv2.putText(self.frame, "Left Side Distance: " + str(right_side_percentage), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 10), 2)
                                    msg = " Face Movement Detected...! stop exam ..!"
                                    self.outer_left_right_counter = 0
                                    self.internal_left_right_counter = 0
                                    self.Stop_Exam(msg)
                                    break
                # show web-cam
                cv2.imshow("Web_cam_on", self.frame)
                escap_key = cv2.waitKey(1)
                # if press Escape char then break loop
                if escap_key == 27:
                    cv2.destroyAllWindows()
                    logger.info(" Video camera Frame released...! ")
                    break

        except ModuleNotFoundError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except SystemError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except InterruptedError as error:
            logger.error(error.__class__.__name__ + ": " + error.message)
        except Exception as exception:
            logger.error(exception.__class__.__name__ + ": " + exception.message)

if __name__ == "__main__":

    FaceMovementDetection_obj = Face_Movement_Detection()
    FaceMovementDetection_obj.Get_Training_Data()
    FaceMovementDetection_obj.Start_Web_Cam()