import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Membuat objek capture video
cap = cv2.VideoCapture(1)  # '0' untuk kamera default

# Mengatur resolusi kamera ke HD 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses")
    exit()

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# mengatur setpoint
setpoint = 1 #isikan 1,2 atau 3


while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat menerima frame (stream end?). Exiting ...")
        break

    # Aplikasi MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Mendapatkan koordinat telapak kaki kiri dan kanan
        kaki_kiri = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        kaki_kanan = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        
        # Menyimpan koordinat dalam variabel
        koordinat_kaki_kiri_x = kaki_kiri.x
        koordinat_kaki_kiri_y = kaki_kiri.y
        koordinat_kaki_kanan_x = kaki_kanan.x
        koordinat_kaki_kanan_y = kaki_kanan.y

        # Menampilkan koordinat pada frame
        cv2.putText(frame, f"Kaki Kiri X: {koordinat_kaki_kiri_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Kaki Kiri Y: {koordinat_kaki_kiri_y:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Kaki Kanan X: {koordinat_kaki_kanan_x:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Kaki Kanan Y: {koordinat_kaki_kanan_y:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Menggambar garis set point untuk ketinggian lompatan
    if setpoint == 1 :
        set_point_atas = 500
        set_point_bawah = 720
    elif setpoint == 2 :
        set_point_atas = 400
        set_point_bawah = 720
    elif setpoint == 3 :
        set_point_atas = 300
        set_point_bawah = 720   
    else : 
        set_point_atas = 200
        set_point_bawah = 500

    cv2.line(frame, (0, set_point_atas), (1280, set_point_atas), (0, 255, 0), 2)
    cv2.line(frame, (0, set_point_bawah), (1280, set_point_bawah), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()