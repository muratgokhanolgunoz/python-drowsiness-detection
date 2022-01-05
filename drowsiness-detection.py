from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import wmi
import pyscreenshot
from datetime import datetime
import os


def eye_aspect_ratio(eye):
    # Creating eye contour coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    # Calculating center point of the eyes
    ear = (A + B) / (2 * C)
    return ear

programs = []
f = wmi.WMI()

# The value of the angle of the eye
thresh = 0.25

# The number of checked frames per second
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("data.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)

        ear = (leftEar + rightEar) / 2

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 2)

        if ear < thresh:
            flag += 1
            # print(flag)
            if flag >= frame_check:
                # To capture the screen
                image = pyscreenshot.grab()

                # To display the captured screenshot
                # image.show()

                # To save the screenshot
                # Get current datetime from filename
                filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                image.save("./screenshots/screenshot-%s.png" % filename)
                # print(f"{datetime.now()} ==> Take screenshot")

                #  Retrieving currently running programs in the operating system
                for process in f.Win32_Process():
                    # os.system(f"TASKKILL /F /IM {process.Name}")
                    # programs.append(process.Name)
                    print(f"{process.ProcessId} => {process.Name}");
                    
                # os.system("shutdown /s /t 1")
                break
        else:
            flag = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("e"):
        break

cv2.destroyAllWindows()
cap.release()
