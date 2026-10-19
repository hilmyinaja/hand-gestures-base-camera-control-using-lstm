import cv2
import time
import os
import platform
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Topology 21 Landmark Tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Daftar 8 Kelas Gestur Tugas Akhir
GESTURE_CLASSES = [
    "rotate_cw",     # 1. Putar telunjuk searah jarum jam (Naikkan Nilai)
    "rotate_ccw",    # 2. Putar telunjuk lawan jarum jam (Turunkan Nilai)
    "next_mode",     # 3. Geser ke atas / Swipe Up (Ganti Parameter)
    "lock_unlock",   # 4. Kepalkan tangan / Fist (Kunci / Buka Kunci)
    "zoom_in",       # 5. Jari mekar keluar / Pinch Out (Zoom In)
    "zoom_out",      # 6. Jari menguncup ke dalam / Pinch In (Zoom Out)
    "snap_photo",    # 7. Pose jepret / Pinch cepat (Ambil Foto)
    "idle"           # 8. Tangan diam / pose rileks netral (Mencegah salah pemicu)
]

# Konfigurasi FPS & Sekuens sesuai performa Raspberry Pi (20 FPS)
TARGET_FPS = 20
SEQUENCE_LENGTH = 20           # 20 frame per sekuens = 1.0 detik durasi gerakan
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 0.05 detik (50 ms) per frame
DATASET_DIR = "dataset"

def ensure_model_exists(model_path="hand_landmarker.task"):
    if not os.path.exists(model_path):
        print(f"[Info] Mengunduh model {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("[Info] Model berhasil diunduh!")

def extract_normalized_landmarks(hand_landmarks):
    """Mengekstraksi 21 koordinat 3D dinormalisasi relatif terhadap wrist (landmark 0)."""
    wrist = hand_landmarks[0]
    features = []
    for lm in hand_landmarks:
        norm_x = lm.x - wrist.x
        norm_y = lm.y - wrist.y
        norm_z = lm.z - wrist.z
        features.extend([norm_x, norm_y, norm_z])
    return np.array(features, dtype=np.float32)

def count_samples(class_name):
    """Menghitung berapa banyak sampel .npy yang sudah tersimpan untuk kelas ini."""
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_dir):
        return 0
    return len([f for f in os.listdir(class_dir) if f.endswith(".npy")])

def open_camera():
    """Membuka webcam langsung tanpa blocking."""
    is_windows = (platform.system() == "Windows")
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2

    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def main():
    print("=======================================================")
    print("   Perekam Dataset Gestur Tugas Akhir (Full GUI)       ")
    print("=======================================================")

    os.makedirs(DATASET_DIR, exist_ok=True)
    for c in GESTURE_CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    ensure_model_exists(model_path)

    # Inisialisasi Detektor MediaPipe
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Buka Webcam langsung di awal (seperti test_hand_tracking.py)
    print("Mengaktifkan webcam...")
    cap = open_camera()
    if not cap.isOpened():
        print("[Error] Webcam tidak terdeteksi! Pastikan kamera terpasang dan tidak dipakai aplikasi lain.")
        return

    print("Webcam berhasil aktif! Jendela visual akan langsung terbuka.")
    print("Semua kontrol dilakukan melalui keyboard pada jendela visual:")
    print("  - Tekan '1' sampai '8' : Langsung pilih kelas gestur")
    print("  - Tekan 'TAB' atau 'N' : Pindah ke kelas berikutnya")
    print("  - Tekan 'SPASI'        : Mulai hitung mundur & rekam 1 sampel")
    print("  - Tekan 'M'            : Toggle Mirror (Cermin) ON/OFF")
    print("  - Tekan 'Q' atau ESC   : Keluar dari program\n")

    current_class_idx = 0
    mirror_view = True

    state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING"
    countdown_start_time = 0
    countdown_seconds = 3
    recorded_frames = []
    last_frame_recorded_time = 0
    status_msg = "Tekan SPASI untuk mulai rekam"
    status_color = (255, 255, 255)

    start_time = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Peringatan] Gagal membaca frame webcam.")
            time.sleep(0.05)
            continue

        if mirror_view:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # Hitung FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        hand_present = bool(result and result.hand_landmarks)
        hand_features = None

        if hand_present:
            hand_landmarks = result.hand_landmarks[0]
            hand_features = extract_normalized_landmarks(hand_landmarks)

            # Gambar visual landmark
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, coords[s], coords[e], (0, 255, 128), 2)
            for idx, pt in enumerate(coords):
                color = (0, 0, 255) if idx == 0 else (255, 100, 0)
                size = 6 if idx == 0 else 4
                cv2.circle(frame, pt, size, color, -1)

        current_class = GESTURE_CLASSES[current_class_idx]
        sample_count = count_samples(current_class)

        # --- State Machine Perekaman ---
        if state == "IDLE":
            pass

        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start_time
            remaining = countdown_seconds - int(elapsed)
            if remaining > 0:
                status_msg = f"SIAP-SIAP: {remaining}..."
                status_color = (0, 255, 255)
                # Angka besar di tengah layar
                cv2.putText(frame, str(remaining), (w // 2 - 35, h // 2 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 6)
            else:
                state = "RECORDING"
                recorded_frames = []
                last_frame_recorded_time = 0

        elif state == "RECORDING":
            if hand_present:
                now_rec = time.time()
                # Kunci sampling rate ke 20 FPS (setiap 50 ms)
                if (now_rec - last_frame_recorded_time) >= FRAME_INTERVAL:
                    recorded_frames.append(hand_features)
                    last_frame_recorded_time = now_rec

                current_len = len(recorded_frames)
                status_msg = f"● MEREKAM (20 FPS): {current_len}/{SEQUENCE_LENGTH}"
                status_color = (0, 0, 255)

                # Progress bar di bawah layar
                progress_w = int((current_len / SEQUENCE_LENGTH) * (w - 40))
                cv2.rectangle(frame, (20, h - 35), (20 + progress_w, h - 20), (0, 0, 255), -1)

                if current_len == SEQUENCE_LENGTH:
                    # Simpan sekuens ke file .npy
                    class_dir = os.path.join(DATASET_DIR, current_class)
                    existing_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
                    next_num = len(existing_files) + 1
                    file_name = f"seq_{next_num:03d}.npy"
                    file_path = os.path.join(class_dir, file_name)

                    np.save(file_path, np.array(recorded_frames))
                    print(f"✔ [{current_class}] Tersimpan: {file_name}")

                    state = "IDLE"
                    status_msg = f"✔ TERSIMPAN ({file_name})! Tekan SPASI lagi"
                    status_color = (0, 255, 0)
            else:
                # Batalkan jika tangan hilang di tengah rekaman
                state = "IDLE"
                recorded_frames = []
                status_msg = "[BATAL] Tangan tidak terdeteksi! Ulangi SPASI"
                status_color = (0, 0, 255)

        # --- Panel Informasi Overlay GUI ---
        # Kotak Header Status
        cv2.rectangle(frame, (10, 10), (460, 115), (25, 25, 25), -1)
        cv2.putText(frame, f"Kelas [{current_class_idx + 1}/8]: {current_class} ({sample_count} sampel)", 
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_msg}", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2)
        cv2.putText(frame, f"FPS: {int(fps)} | Tekan 1-8 utk pilih kelas | TAB: Next", (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Daftar Mini Kelas di sisi kanan atas
        menu_y = 25
        for i, c_name in enumerate(GESTURE_CLASSES):
            c_color = (0, 255, 128) if i == current_class_idx else (160, 160, 160)
            c_count = count_samples(c_name)
            cv2.putText(frame, f"{i+1}.{c_name} ({c_count})", (w - 170, menu_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_color, 1)
            menu_y += 18

        cv2.imshow("Data Collector - Tugas Akhir Hilmy", frame)

        # --- Penanganan Input Tombol Keyboard ---
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Tombol SPASI
            if state == "IDLE":
                state = "COUNTDOWN"
                countdown_start_time = time.time()

        elif ord('1') <= key <= ord('8'):
            # Langsung pilih kelas dengan tombol angka 1-8
            current_class_idx = key - ord('1')
            state = "IDLE"
            status_msg = f"Beralih ke kelas {GESTURE_CLASSES[current_class_idx]}"
            status_color = (255, 255, 255)

        elif key == 9 or key == ord('n') or key == ord('N'):  # Tombol TAB atau 'N'
            # Pindah ke kelas berikutnya
            current_class_idx = (current_class_idx + 1) % len(GESTURE_CLASSES)
            state = "IDLE"
            status_msg = f"Beralih ke kelas {GESTURE_CLASSES[current_class_idx]}"
            status_color = (255, 255, 255)

        elif key == ord('b') or key == ord('B'):
            # Pindah ke kelas sebelumnya
            current_class_idx = (current_class_idx - 1) % len(GESTURE_CLASSES)
            state = "IDLE"

        elif key == ord('m') or key == ord('M'):
            mirror_view = not mirror_view

        elif key == ord('q') or key == 27:  # Tombol Q atau ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        detector.close()
    except:
        pass
    print("\nProgram selesai. Sampel kamu tersimpan di folder 'dataset/'.")

if __name__ == "__main__":
    main()
