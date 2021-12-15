from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
    # Göz çevresi koordinatları oluşturuluyor
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3]) 

    # Gözün orta noktası hesaplanıyor       
	ear = (A + B) / (2 * C)      
	return ear
	
# Gözün açı değeri
thresh = 0.25

# Saniye başına kontrol edilen kare sayısı
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("data.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width = 700)
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
			print (flag)
			if flag >= frame_check:				
				cv2.putText(frame, "W A R N I N G", (10, 500),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   
		else:
			flag = 0
    
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("e"):
		break
    
cv2.destroyAllWindows()
cap.release() 

