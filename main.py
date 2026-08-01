import cv2
from ultralytics import YOLO
from datetime import datetime
import time
import winsound
import csv
import os
import smtplib
from email.message import EmailMessage
screenshot_saved = False

#YOLO model load
model = YOLO("yolov8n.pt")

#camera start
cap = cv2.VideoCapture(0)
def send_email(image_path):
    sender_email = "YOUR_EMAIL@gmail.com"
    sender_password = "YOUR_APP_PASSWORD"
    receiver_email = "RECEIVER_EMAIL@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = "Road Accident Detected"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content("Accident detected.\nPlease check the attached screenshot.")
    with open(image_path,"rb") as f:
        msg.add_attachement(
            f.read(),
            maintype ="image",
            subtype ="jpeg",
            filename ="image_path"
                            )
        
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:
     smtp.login(sender_email,sender_password)
    smtp.send_message(msg)

while True:
    ret,frame = cap.read()

    if not ret:
        break

    #Detection
    results = model(frame,
    classes = [2,3,5,7])
    annotated_frame = results[0].plot()
    
    vehicles = len(results[0].boxes)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int,box.xyxy[0])
        confidence = float(box.conf[0])
        cv2.putText(annotated_frame, f"{confidence:.2f}",
                    (x1,y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,0),
                    2
        )
        if vehicles >=2:
            cv2.rectangle(annotated_frame, (x1, y1),(x2, y2),(0,0,225),3)
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])

        if model.names[cls] == "car":
            x1, y1, x2, y2 = map(int,box.xyxy[0])
            print("Car Position:",x1, y1, x2, y2)

    #Draw boxes
    annotated_frame = results[0].plot()
    vehicles = len(results[0].boxes)
    accident = False
    
    boxes = results[0].boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int,boxes[i].xyxy[0])
        for j in range(i + 1,len(boxes)):
            a1, b1, a2, b2 = map(int, boxes[j].xyxy[0])
            if abs(x1 - a1) < 80 and abs(y1 - b1) < 80:
                accident = True
    if accident:
        cv2.putText(annotated_frame,"ACCIDENT DETECTED!",
                    (40,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
        )
    vehicles = len(results[0].boxes)
    print("Accident =", accident)
    print("Screenshot =",screenshot_saved)
    if vehicles >= 2:
        winsound.Beep(1000, 500)
        
        latitude = 28.6692
        longitude = 77.4538
        print(f"GPS Location: {latitude},{longitude}")

        
                    
        filename = f"Accident_{int(time.time())}.jpg"
        cv2.imwrite(filename,annotated_frame)
        screenshot_saved = True
        print("Screenshot saved:",filename)

    
    
        cv2.putText(annotated_frame,f"GPS: {latitude},{longitude}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                    )
        cv2.imwrite(filename,annotated_frame)
        file_exists = os.path.isfile(filename)
        with open("accident_log.csv", "a", newline="") as file:
             writer = csv.writer(file)

             if not file_exists:
                    writer.writerow(["Date", "Time", "Latitude", "Longitude", "Image", "Vehicles"])
                                      
             writer.writerow([datetime.now().strftime("%d-%m-%Y"),datetime.now().strftime("%H:%M:%S"),latitude, longitude, filename, vehicles])
             print("screenshot saved:",filename)
             print("Vehicles Detected:",vehicles)
             cv2.putText(
            annotated_frame,
            "WARNING:Possible Accident!",
            (20,80),
           cv2.FONT_HERSHEY_SIMPLEX,
            1,
           (0,0,255),
            3
        )
    cv2.putText(annotated_frame,f"Vehicles:{vehicles}",
                (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,225,0),
    2
    )
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(annotated_frame,current_time,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
                )



    #Show result
    cv2.imshow("Road Accident Detector",annotated_frame)

    #press q to exit
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()