import cv2
import mediapipe as mp

vid = cv2.VideoCapture(0)

mediaPipe_hands = mp.solutions.hands
mediaPipe_drawing = mp.solutions.drawing_utils

hands = mediaPipe_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tip_ids = [4,8,12,16,20]

def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        land_mark = hand_landmarks[handNo].landmark
        print(land_mark)
        fingers = []
        for lm in tip_ids:
            if lm != 4:
                finger_tip_x = land_mark[lm].x
                finger_bottom_x = land_mark[lm-2].x
                index_tip_y = land_mark[8].y
                thumb_tip_y = land_mark[4].y
                if finger_tip_x < finger_bottom_x:
                    fingers.append(1)
                if finger_tip_x > finger_bottom_x:
                    fingers.append(0)
                total_fingers = fingers.count(1)
                if total_fingers == 0:
                    if index_tip_y > thumb_tip_y:
                        like = "Like"
                        cv2.putText(image,like, (50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,255,0),4)
                    if index_tip_y < thumb_tip_y:
                        dislike = "Dislike"
                        cv2.putText(image,dislike, (50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,0,255),4)

def drawHandLanmarks(image, hand_landmarks):
    if hand_landmarks:
        for lm in hand_landmarks:
            mediaPipe_drawing.draw_landmarks(image, lm, mediaPipe_hands.HAND_CONNECTIONS)



while True:
    ret, image = vid.read()
    image = cv2.flip(image,1)

    results = hands.process(image)

    hand_landmarks = results.multi_hand_landmarks

    drawHandLanmarks(image, hand_landmarks)
    countFingers(image,hand_landmarks)

    cv2.imshow("Like, or Dislike", image)
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()