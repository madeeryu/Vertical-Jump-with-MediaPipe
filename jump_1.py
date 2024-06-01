import cv2
import mediapipe as mp
import csv
import time  # Impor modul time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Membuat objek capture video
cap = cv2.VideoCapture(1)  # '0' untuk kamera laptop ku '0' kamera hp ku


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses")
    exit()

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

nilai_tertinggi_kaki_kiri = 0
nilai_tertinggi_kaki_kanan = 0

def tulis_ke_csv(nilai_kiri, nilai_kanan):
    with open('nilai_tertinggi.csv', 'w', newline='') as file:  # Gunakan mode 'w' untuk menulis ulang file
        writer = csv.writer(file)
        writer.writerow([nilai_kiri, nilai_kanan])

# set point
set_point_rendah = 600
set_point_sedang = 500
set_point_tinggi = 400

kondisi = "rendah"  # Opsi: 'rendah', 'sedang', 'tinggi'


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
        
        # Mengkonversi koordinat relatif ke koordinat piksel
        h, w, _ = frame.shape
        kaki_kiri_x = int(kaki_kiri.x * w)
        kaki_kiri_y = int(kaki_kiri.y * h)
        kaki_kanan_x = int(kaki_kanan.x * w)
        kaki_kanan_y = int(kaki_kanan.y * h)

        # Menentukan set point berdasarkan kondisi
        if kondisi == "rendah":
            set_point = set_point_rendah
        elif kondisi == "sedang":
            set_point = set_point_sedang
        else:
            set_point = set_point_tinggi

        # Menggambar garis set point untuk ketinggian lompatan
        cv2.line(frame, (0, set_point), (1280, set_point), (0, 255, 0), 2)

        # Menggambar garis yang mengikuti tracking lompatan pada sumbu x
        if kaki_kiri_y < set_point or kaki_kanan_y < set_point:
            min_y = min(kaki_kiri_y, kaki_kanan_y)
            cv2.line(frame, (0, min_y), (1280, min_y), (255, 0, 0), 2)

            # Menggambar garis yang mengikuti tracking lompatan pada sumbu y jika di atas set point
            if kaki_kiri_y < set_point:
                cv2.line(frame, (kaki_kiri_x, min_y), (kaki_kiri_x, set_point), (255, 0, 0), 2)
            if kaki_kanan_y < set_point:
                cv2.line(frame, (kaki_kanan_x, min_y), (kaki_kanan_x, set_point), (255, 0, 0), 2)

            # Menghitung jarak dari set point ke posisi kaki pada sumbu y
            jarak_kaki_kiri = abs(set_point - kaki_kiri_y)
            jarak_kaki_kanan = abs(set_point - kaki_kanan_y)
            
            # Menampilkan jarak pada frame
            cv2.putText(frame, f"Jarak Kaki Kiri: {jarak_kaki_kiri} px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Jarak Kaki Kanan: {jarak_kaki_kanan} px", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Perbarui nilai tertinggi dan tulis ke CSV jika ada nilai baru yang lebih tinggi
            update_terjadi = False
            if jarak_kaki_kiri > nilai_tertinggi_kaki_kiri:
                nilai_tertinggi_kaki_kiri = jarak_kaki_kiri
                update_terjadi = True
            
            if jarak_kaki_kanan > nilai_tertinggi_kaki_kanan:
                nilai_tertinggi_kaki_kanan = jarak_kaki_kanan
                update_terjadi = True

            if update_terjadi:
                tulis_ke_csv(nilai_tertinggi_kaki_kiri, nilai_tertinggi_kaki_kanan)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()