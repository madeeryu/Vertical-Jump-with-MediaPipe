from controller import Robot, Camera, Motor
import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Membuat instance robot
robot = Robot()

# Mendapatkan time step dari world saat ini
timestep = int(robot.getBasicTimeStep())

# Mengaktifkan kamera
camera = robot.getDevice('camera')
camera.enable(timestep)

# Mengatur motor - set ke default (kecepatan = 0)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# List untuk menyimpan data kecepatan
motor_kiri_data = []
motor_kanan_data = []

# Set error
e_prev = 0

# Skala digunakan untuk Mengubah Ukuran Jendela OpenCV
scale = 0.65

def calculate_motor(parameter):
    return (parameter / 10) * 6.28

# Persiapan plot
fig, ax2 = plt.subplots(figsize=(8, 4))

# Main loop:
# - melakukan langkah simulasi sampai Webots menghentikan controller
while robot.step(timestep) != -1:
    # Definisi fungsi keanggotaan
    # Sensor range
    x_eror = np.arange(-640, 640, 0.1)
    x_delta_eror = np.arange(-640, 640, 0.1)

    # Motor range
    x_kategori_motor = np.arange(-7, 7, 0.01)

    # Fungsi keanggotaan untuk error kamera
    eror_terlalu_kecil = fuzz.trapmf(x_eror, [-640, -640, -320, -200])
    eror_kecil = fuzz.trimf(x_eror, [-320, -180, -40])
    eror_zero = fuzz.trapmf(x_eror, [-60, -40, 40, 60])
    eror_besar = fuzz.trimf(x_eror, [40, 180, 320])
    eror_terlalu_besar = fuzz.trapmf(x_eror, [200, 320, 640, 640])

    # Fungsi keanggotaan untuk delta error
    delta_terlalu_kecil = fuzz.trapmf(x_delta_eror, [-640, -640, -320, -200])
    delta_kecil = fuzz.trimf(x_delta_eror, [-320, -180, -40])
    delta_zero = fuzz.trapmf(x_delta_eror, [-60, -40, 40, 60])
    delta_besar = fuzz.trimf(x_delta_eror, [40, 180, 320])
    delta_terlalu_besar = fuzz.trapmf(x_delta_eror, [200, 320, 640, 640])

    # Fungsi keanggotaan untuk kecepatan motor
    motor_paling_kiri_mf = fuzz.trapmf(x_kategori_motor, [-7, -7, -6, -5])
    motor_kiri_mf = fuzz.trapmf(x_kategori_motor, [-6, -5, -4, -3])
    motor_agak_kiri_mf = fuzz.trapmf(x_kategori_motor, [-4, -3, -2, -1])
    motor_lurus_mf = fuzz.trapmf(x_kategori_motor, [-2, -1, 1, 2])
    motor_agak_kanan_mf = fuzz.trapmf(x_kategori_motor, [1, 2, 3, 4])
    motor_kanan_mf = fuzz.trapmf(x_kategori_motor, [3, 4, 5, 6])
    motor_paling_kanan_mf = fuzz.trapmf(x_kategori_motor, [5, 6, 7, 7])

    # Mengakses kamera
    image = camera.getImage()
    img_np = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img_np = img_np[:, :, :3]  # Membuang channel alpha
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Mengubah ke RGB

    # Mengubah ke grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    gray_resized = cv2.resize(gray, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))

    # Menyaring warna untuk membuat binary mask
    _, black_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    black_mask_resized = cv2.resize(black_mask, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))

    # Menggambar lingkaran set point
    x_pot = int(width / 2)
    y_pot = int(height / 2)
    cv2.circle(img_np, (x_pot, y_pot), 10, (0, 0, 255), -1)

    # Menemukan kontur dari black mask
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Mendapatkan kontur terbesar
        biggest_contour = max(contours, key=cv2.contourArea)

        # Mendapatkan bounding rectangle
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Menggambar bounding rectangle
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mendapatkan pusat dari bounding rectangle
        center_x = x + w // 2
        center_y = y_pot

        # Menggambar pusat dari bounding rectangle
        cv2.circle(img_np, (center_x, center_y), 5, (255, 0, 0), -1)

        # Menggambar garis dari pusat bounding rectangle ke pusat gambar
        cv2.line(img_np, (center_x, center_y), (x_pot, y_pot), (255, 255, 0), 2)

        # Mendapatkan waktu dunia
        time = robot.getTime()

        # Mendapatkan error (Set point dari kamera - Pusat bounding rectangle)
        error = x_pot - center_x

        # Mendapatkan delta error
        delta_err = error - e_prev

        # Mengatur error saat ini sebagai error sebelumnya untuk iterasi berikutnya
        e_prev = error

        # Compute memberships
        etki = fuzz.interp_membership(x_eror, eror_terlalu_kecil, error)
        eki = fuzz.interp_membership(x_eror, eror_kecil, error)
        ez = fuzz.interp_membership(x_eror, eror_zero, error)
        eka = fuzz.interp_membership(x_eror, eror_besar, error)
        etka = fuzz.interp_membership(x_eror, eror_terlalu_besar, error)

        dtki = fuzz.interp_membership(x_delta_eror, delta_terlalu_kecil, delta_err)
        dki = fuzz.interp_membership(x_delta_eror, delta_kecil, delta_err)
        dz = fuzz.interp_membership(x_delta_eror, delta_zero, delta_err)
        dka = fuzz.interp_membership(x_delta_eror, delta_besar, delta_err)
        dtka = fuzz.interp_membership(x_delta_eror, delta_terlalu_besar, delta_err)

        # Fuzzy rules
        #ETKI (eror terlalu kiri), EKI (eror kiri), EZ (eror zero), EKA (eror kanan), ETKA (eror terlalu kanan)
        #DTKI (deltaeror terlalu kiri), DKI (deltaeror kiri), DZ (deltaeror zero), DKA (deltaeror kanan), DTKA (deltaeror terlalu kanan)
        rules_1 = np.fmin(etki, dtki)
        rules_2 = np.fmin(etki, dki)
        rules_3 = np.fmin(etki, dz)
        rules_4 = np.fmin(etki, dka)
        rules_5 = np.fmin(etki, dtka)

        rules_6 = np.fmin(eki, dtki)
        rules_7 = np.fmin(eki, dki)
        rules_8 = np.fmin(eki, dz)
        rules_9 = np.fmin(eki, dka)
        rules_10 = np.fmin(eki, dtka)

        rules_11 = np.fmin(ez, dtki)
        rules_12 = np.fmin(ez, dki)
        rules_13 = np.fmin(ez, dz)
        rules_14 = np.fmin(ez, dka)
        rules_15 = np.fmin(ez, dtka)

        rules_16 = np.fmin(eka, dtki)
        rules_17 = np.fmin(eka, dki)
        rules_18 = np.fmin(eka, dz)
        rules_19 = np.fmin(eka, dka)
        rules_20 = np.fmin(eka, dtka)

        rules_21 = np.fmin(etka, dtki)
        rules_22 = np.fmin(etka, dki)
        rules_23 = np.fmin(etka, dz)
        rules_24 = np.fmin(etka, dka)
        rules_25 = np.fmin(etka, dtka)

        # Persentase_paling_kiri =  np.fmin(rules_1, motor_paling_kiri_mf )
        # Persentase_kiri =  np.fmin(np.fmax(rules_2, np.fmax(rules_6, np.fmax(rules_11, rules_16))), motor_kiri_mf )
        # Persentase_agak_kiri =  np.fmin(np.fmax(rules_3, np.fmax(rules_7, np.fmax(rules_12, np.fmax( rules_17, rules_21)))), motor_agak_kiri_mf )
        # Persentase_lurus =  np.fmin(np.fmax(rules_4, np.fmax(rules_8, np.fmax(rules_13, np.fmax(rules_18, rules_22)))), motor_lurus_mf )
        # Persentase_agak_kanan =  np.fmin(np.fmax(rules_5, np.fmax(rules_9, np.fmax(rules_14, np.fmax(rules_19, rules_23)))), motor_agak_kanan_mf )
        # Persentase_kanan =  np.fmin(np.fmax (rules_10, np.fmax(rules_15, np.fmax(rules_18, np.fmax(rules_20, rules_24)))), motor_kanan_mf )
        # Persentase_paling_kanan =  np.fmin(np.fmax(rules_15, np.fmax(rules_19, np.fmax(rules_20,np.fmax(rules_23, rules_25)))), motor_paling_kanan_mf )

        Persentase_paling_kiri =  np.fmin(rules_1, motor_paling_kiri_mf )
        Persentase_kiri =  np.fmin(np.fmax(rules_2, np.fmax(rules_6, np.fmax(rules_11, rules_16))), motor_kiri_mf )
        Persentase_agak_kiri =  np.fmin(np.fmax(rules_3, np.fmax(rules_7, np.fmax(rules_12, np.fmax( rules_17, rules_21)))), motor_agak_kiri_mf )
        Persentase_lurus =  np.fmin(np.fmax(rules_4, np.fmax(rules_8, np.fmax(rules_13, np.fmax(rules_18, rules_22)))), motor_lurus_mf )
        Persentase_agak_kanan =  np.fmin(np.fmax(rules_5, np.fmax(rules_9, np.fmax(rules_14, np.fmax(rules_19, rules_23)))), motor_agak_kanan_mf )
        Persentase_kanan =  np.fmin(np.fmax (rules_10, np.fmax(rules_15, np.fmax(rules_20, rules_24))), motor_kanan_mf )
        Persentase_paling_kanan =  np.fmin(rules_25, motor_paling_kanan_mf )

        hasil_motor = np.zeros_like(x_kategori_motor)

        # Defuzzifikasi kanan
        aggregated_motor = np.fmax(Persentase_paling_kiri, np.fmax(Persentase_kiri, np.fmax(Persentase_agak_kiri, np.fmax(Persentase_lurus, np.fmax(Persentase_agak_kanan, np.fmax(Persentase_kanan, Persentase_paling_kanan))))))
        kecepatan_motor = fuzz.defuzz(x_kategori_motor, aggregated_motor, 'centroid')
        hasil_vic = fuzz.interp_membership(x_kategori_motor, aggregated_motor, kecepatan_motor)

        # Mengatur Kecepatan Motor dan Menampilkan Gambar
        t = robot.getTime()

        if t == 0.032:
            delta_err = 0

        motor_fix = calculate_motor(kecepatan_motor)
        
        motor_kiri = 2 - motor_fix
        motor_kanan = 2 + motor_fix

        if motor_kiri >= 6.28:
            motor_kiri = 6.28
        elif motor_kanan >= 6.28:
            motor_kanan = 6.28

        print(f"Motor Kiri: {motor_kiri:.2f} || Motor Kanan: {motor_kanan:.2f} || Eror: {error} || Delta Eror: {delta_err}")

        leftMotor.setVelocity(motor_kiri)
        rightMotor.setVelocity(motor_kanan)

        # Simpan data kecepatan untuk plot
        motor_kiri_data.append(motor_kiri)
        motor_kanan_data.append(motor_kanan)

        # Show the OpenCV Window
        img_np_resized = cv2.resize(img_np, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)))
        #cv2.imshow("Gray", gray_resized)
        #cv2.imshow("Black Mask", black_mask_resized)
        cv2.imshow("Webots Camera", img_np_resized)

    # Exit the OpenCV Windows using 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menggunakan moving average untuk smooth data
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 5
motor_kiri_smooth = moving_average(motor_kiri_data, window_size)
motor_kanan_smooth = moving_average(motor_kanan_data, window_size)

# Plotting data kecepatan motor
plt.figure(figsize=(12, 6))
# plt.plot(motor_kiri_data, label='Kecepatan Motor Kiri (Original)')
plt.plot(range(window_size-1, len(motor_kiri_data)), motor_kiri_smooth, label='Kecepatan Motor Kiri (Smoothed)')
# plt.plot(motor_kanan_data, label='Kecepatan Motor Kanan (Original)')
plt.plot(range(window_size-1, len(motor_kanan_data)), motor_kanan_smooth, label='Kecepatan Motor Kanan (Smoothed)')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Kecepatan Motor')
plt.title('Grafik Kecepatan Motor')
plt.show()

# Close OpenCV Window when the robot restarts
cv2.destroyAllWindows()
plt.close()