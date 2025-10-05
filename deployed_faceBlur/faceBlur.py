import cv2
import mediapipe as mp
from cvzone.FaceDetectionModule import  FaceDetector



def faceBlurrer(inputPath , outputPath,detectionConf = 0.7,kernal = 35):

    detector = FaceDetector(minDetectionCon=detectionConf)
    cap = cv2.VideoCapture(inputPath)

    four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output = cv2.VideoWriter(outputPath,four_cc,fps,(width,height))

    while True:
        Rendered , frame = cap.read()
        if not Rendered: break
        
        faceImg,bboxs = detector.findFaces(frame,draw=False)

        for i,bbox in enumerate(bboxs):
            x , y , w , h = bbox['bbox']

            if x < 0 : x = 0
            if y < 0 : y = 0

            faceCropped = frame[y:y+h,x:x+w]
            blurred = cv2.blur(faceCropped,(kernal,kernal))
            frame[y:y+h,x:x+w] = blurred

        output.write(frame)

    cap.release()
    output.release()

