import cv2
import time
import math
import dlib
import random
from threading import Thread
from playsound import playsound
import customtkinter as ctk
from PIL import Image, ImageTk

# Constants
CALIBRATION_TIME = 5
YAWN_COOLDOWN = 3
BREAK_INTERVAL = 2700  # 45 minutes
MOTIVATION_INTERVAL = 60  # seconds
GRACE_PERIOD = 50

# Audio files
EYES_CLOSE_SOUND = "Eyes Close.mp3"
YAWN_SOUND = "While Yawn.mp3"
DISTRACTED_SOUND = "Distracted.mp3"
BREAK_SOUND = "Break.mp3"

# Initialize detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global state variables
calibrated = False
sound_played = False
face_lost_count = 0
last_ear_above_threshold_time = time.time()
last_yawn_time = 0
last_break_alert_time = time.time()
last_motivation_time = time.time()
study_timer_start = time.time()
ear_values = []
dynamic_ear_threshold = 0.2
motivational_messages = [
    "Keep going, you're doing great!",
    "Stay sharp, you got this!",
    "One step at a time, you're winning!",
    "Keep your eyes on the goal!",
    "Hard work always pays off!"
]

def euclidean(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = euclidean(mouth[13], mouth[19])
    B = euclidean(mouth[14], mouth[18])
    C = euclidean(mouth[12], mouth[16])
    return (A + B) / (2.0 * C)

def play_sound(file):
    try:
        Thread(target=playsound, args=(file,), daemon=True).start()
    except Exception:
        pass

# GUI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("FocusPal - AI Study Assistant")
root.geometry("900x720")

video_label = ctk.CTkLabel(root, text="", width=800, height=600)
video_label.pack(pady=10)

info_label = ctk.CTkLabel(root, text="Initializing...", font=ctk.CTkFont(size=16))
info_label.pack(pady=10)

presence_status = ctk.CTkLabel(root, text="Presence: Unknown", font=ctk.CTkFont(size=14))
presence_status.pack()

eye_status = ctk.CTkLabel(root, text="Eyes: Unknown", font=ctk.CTkFont(size=14))
eye_status.pack()

yawn_status = ctk.CTkLabel(root, text="Yawning: Unknown", font=ctk.CTkFont(size=14))
yawn_status.pack()

status_label = ctk.CTkLabel(root, text="", font=ctk.CTkFont(size=14))
status_label.pack(pady=5)

cap = cv2.VideoCapture(0)

def update_frame():
    global calibrated, face_lost_count, sound_played, last_ear_above_threshold_time
    global last_yawn_time, last_break_alert_time, last_motivation_time, study_timer_start
    global dynamic_ear_threshold, ear_values

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    face_detected = len(faces) > 0

    if face_detected:
        presence_status.configure(text="Presence: Detected")
        face_lost_count = 0
        sound_played = False
    else:
        presence_status.configure(text="Presence: Not Detected")
        face_lost_count += 1

    if face_lost_count > GRACE_PERIOD and not sound_played:
        play_sound(DISTRACTED_SOUND)
        sound_played = True

    eyes_closed_flag = False
    yawn_flag = False

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray, rect)

        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

        if not calibrated:
            if time.time() - study_timer_start < CALIBRATION_TIME:
                ear_values.append(ear)
            else:
                if ear_values:
                    avg_ear = sum(ear_values) / len(ear_values)
                    dynamic_ear_threshold = avg_ear * 0.75
                calibrated = True

        if calibrated:
            if ear < dynamic_ear_threshold:
                eye_status.configure(text="Eyes: Closed")
                eyes_closed_flag = True
                if time.time() - last_ear_above_threshold_time > 5:
                    play_sound(EYES_CLOSE_SOUND)
                    last_ear_above_threshold_time = time.time()
            else:
                eye_status.configure(text="Eyes: Open")
                last_ear_above_threshold_time = time.time()

        mouth = [landmarks.part(i) for i in range(48, 68)]
        mar = calculate_mar(mouth)
        if mar > 0.5 and time.time() - last_yawn_time > YAWN_COOLDOWN:
            yawn_flag = True
            play_sound(YAWN_SOUND)
            last_yawn_time = time.time()

    yawn_status.configure(text=f"Yawning: {'Yes' if yawn_flag else 'No'}")

    current_time = time.time() - study_timer_start
    minutes, seconds = divmod(int(current_time), 60)
    info_label.configure(text=f"Study Time: {minutes}m {seconds}s")

    if time.time() - last_break_alert_time > BREAK_INTERVAL:
        play_sound(BREAK_SOUND)
        last_break_alert_time = time.time()

    if time.time() - last_motivation_time > MOTIVATION_INTERVAL:
        motivational_message = random.choice(motivational_messages)
        status_label.configure(text=motivational_message)
        last_motivation_time = time.time()

    # Resize frame to 800x600 and convert to ImageTk.PhotoImage for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (800, 600))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
