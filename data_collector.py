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

# Daftar 9 Kelas Gestur Tugas Akhir
GESTURE_CLASSES = [
    "rotate_cw",     # 1. Putar telunjuk searah jarum jam (Naikkan Nilai)
    "rotate_ccw",    # 2. Putar telunjuk lawan jarum jam (Turunkan Nilai)
    "next_mode",     # 3. Geser ke atas / Swipe Up (Ganti Parameter)
    "lock_unlock",   # 4. Kepalkan tangan / Fist (Kunci / Buka Kunci)
    "zoom_in",       # 5. Jari mekar keluar / Pinch Out (Zoom In)
    "zoom_out",      # 6. Jari menguncup ke dalam / Pinch In (Zoom Out)
    "snap_photo",    # 7. Pose jepret / Pinch cepat (Ambil Foto)
    "record_video",  # 8. Pose dua jari / Peace / V-Sign ✌️ (Mulai/Stop Video)
    "idle"           # 9. Tangan diam / pose rileks netral (Mencegah salah pemicu)
]

# Konfigurasi FPS & Sekuens sesuai performa Raspberry Pi (20 FPS)
TARGET_FPS = 20
SEQUENCE_LENGTH = 20           # 20 frame per sekuens = tepat 1.0 detik durasi gerakan
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 0.05 detik (50 ms) per frame
BURST_REST_DURATION = 1.2      # 1.2 detik jeda istirahat antar-sampel pada mode Auto-Burst

DATASET_DIR = "dataset"
RAW_VIDEOS_DIR = "raw_videos"

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
    """Menghitung berapa banyak sampel .npy dan .mp4 yang tersimpan untuk kelas ini."""
    npy_dir = os.path.join(DATASET_DIR, class_name)
    mp4_dir = os.path.join(RAW_VIDEOS_DIR, class_name)
    npy_count = len([f for f in os.listdir(npy_dir) if f.endswith(".npy")]) if os.path.exists(npy_dir) else 0
    mp4_count = len([f for f in os.listdir(mp4_dir) if f.endswith(".mp4")]) if os.path.exists(mp4_dir) else 0
    return npy_count, mp4_count

def get_next_sample_index(class_name):
    """Menghitung nomor file berikutnya dengan memeriksa folder dataset dan raw_videos agar sinkron."""
    npy_dir = os.path.join(DATASET_DIR, class_name)
    mp4_dir = os.path.join(RAW_VIDEOS_DIR, class_name)
    
    nums = []
    for d in [npy_dir, mp4_dir]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith("seq_") and (f.endswith(".npy") or f.endswith(".mp4")):
                    try:
                        num = int(f.split("_")[1].split(".")[0])
                        nums.append(num)
                    except (ValueError, IndexError):
                        pass
    return (max(nums) + 1) if nums else 1

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

def save_sequence_data(class_name, sample_idx, recorded_features, recorded_raw_frames, frame_size):
    """Menyimpan array landmark (.npy) DAN video mentah 20 FPS (.mp4) secara sinkron 1:1."""
    class_npy_dir = os.path.join(DATASET_DIR, class_name)
    class_mp4_dir = os.path.join(RAW_VIDEOS_DIR, class_name)
    os.makedirs(class_npy_dir, exist_ok=True)
    os.makedirs(class_mp4_dir, exist_ok=True)

    base_name = f"seq_{sample_idx:03d}"
    npy_path = os.path.join(class_npy_dir, f"{base_name}.npy")
    mp4_path = os.path.join(class_mp4_dir, f"{base_name}.mp4")

    # 1. Simpan Fitur Landmark (20, 63)
    np.save(npy_path, np.array(recorded_features, dtype=np.float32))

    # 2. Simpan Video Mentah 20 FPS (Tepat 20 Frame, 1.0 Detik)
    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(mp4_path, fourcc, float(TARGET_FPS), (w, h))
    for f in recorded_raw_frames:
        writer.write(f)
    writer.release()

    print(f"✔ [{class_name}] Tersimpan: {base_name}.npy & {base_name}.mp4 ({len(recorded_raw_frames)} frames @ 20 FPS)")
    return base_name

def delete_last_sample(class_name):
    """Menghapus sampel terakhir (baik .npy maupun .mp4) agar data tetap sinkron."""
    npy_dir = os.path.join(DATASET_DIR, class_name)
    mp4_dir = os.path.join(RAW_VIDEOS_DIR, class_name)

    sample_nums = []
    for d, ext in [(npy_dir, ".npy"), (mp4_dir, ".mp4")]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith("seq_") and f.endswith(ext):
                    try:
                        num = int(f.split("_")[1].split(".")[0])
                        sample_nums.append(num)
                    except:
                        pass
    if not sample_nums:
        return None

    last_num = max(sample_nums)
    base_name = f"seq_{last_num:03d}"
    deleted = []

    npy_file = os.path.join(npy_dir, f"{base_name}.npy")
    if os.path.exists(npy_file):
        os.remove(npy_file)
        deleted.append(f"{base_name}.npy")

    mp4_file = os.path.join(mp4_dir, f"{base_name}.mp4")
    if os.path.exists(mp4_file):
        os.remove(mp4_file)
        deleted.append(f"{base_name}.mp4")

    return ", ".join(deleted)

def main():
    print("=================================================================")
    print("   Perekam Dataset Gestur Tugas Akhir Hilmy (Npy + Video MP4)   ")
    print("=================================================================")

    # Buat direktori penyimpanan
    for c in GESTURE_CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(RAW_VIDEOS_DIR, c), exist_ok=True)

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

    # Aktifkan Webcam
    print("Mengaktifkan webcam...")
    cap = open_camera()
    if not cap.isOpened():
        print("[Error] Webcam tidak terdeteksi! Pastikan kamera terpasang dan tidak dipakai aplikasi lain.")
        return

    print("Webcam berhasil aktif! Jendela visual terbuka.")
    print("Petunjuk Kontrol Keyboard:")
    print("  - [SPASI]        : Rekam 1 sampel tunggal (Single Shot)")
    print("  - [C]            : Toggle Mode AUTO-BURST (Rekam otomatis berulang)")
    print("  - [1] s/d [9]    : Langsung pilih kelas gestur")
    print("  - [TAB] / [N]    : Pindah ke kelas berikutnya")
    print("  - [B]            : Pindah ke kelas sebelumnya")
    print("  - [D] / [BkSp]   : Hapus sampel terakhir (.npy + .mp4)")
    print("  - [M]            : Toggle Mirror (Cermin) ON/OFF")
    print("  - [Q] / [ESC]    : Keluar dari program\n")

    current_class_idx = 0
    mirror_view = True

    # State Machine
    state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING", "BURST_REST"
    burst_mode = False
    burst_count_session = 0

    countdown_start_time = 0
    countdown_seconds = 3

    recorded_features = []
    recorded_raw_frames = []
    last_frame_recorded_time = 0
    burst_rest_start_time = 0

    status_msg = "Tekan SPASI (Single) atau C (Auto-Burst)"
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

        # Salin frame mentah bersih (TANPA garis skeleton) untuk disimpan ke video .mp4
        clean_raw_frame = frame.copy()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # Hitung FPS real-time
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        hand_present = bool(result and result.hand_landmarks)
        hand_features = None

        if hand_present:
            hand_landmarks = result.hand_landmarks[0]
            hand_features = extract_normalized_landmarks(hand_landmarks)

            # Gambar visual landmark pada frame preview
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, coords[s], coords[e], (0, 255, 128), 2)
            for idx, pt in enumerate(coords):
                color = (0, 0, 255) if idx == 0 else (255, 100, 0)
                size = 6 if idx == 0 else 4
                cv2.circle(frame, pt, size, color, -1)

        current_class = GESTURE_CLASSES[current_class_idx]
        npy_cnt, mp4_cnt = count_samples(current_class)

        # --- Logika State Machine Perekaman ---
        if state == "IDLE":
            pass

        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start_time
            remaining = countdown_seconds - int(elapsed)
            if remaining > 0:
                mode_tag = "[AUTO-BURST]" if burst_mode else "[SINGLE]"
                status_msg = f"{mode_tag} SIAP-SIAP DALAM: {remaining}..."
                status_color = (0, 255, 255)
                # Tampilkan angka besar di tengah layar
                cv2.putText(frame, str(remaining), (w // 2 - 35, h // 2 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 6)
            else:
                state = "RECORDING"
                recorded_features = []
                recorded_raw_frames = []
                last_frame_recorded_time = 0

        elif state == "RECORDING":
            if hand_present:
                now_rec = time.time()
                # Kunci sampling rate ke tepat 20 FPS (interval 50 ms)
                if (now_rec - last_frame_recorded_time) >= FRAME_INTERVAL:
                    recorded_features.append(hand_features)
                    recorded_raw_frames.append(clean_raw_frame)
                    last_frame_recorded_time = now_rec

                current_len = len(recorded_features)
                mode_label = f"[BURST #{burst_count_session + 1}]" if burst_mode else "[SINGLE]"
                status_msg = f"● {mode_label} MEREKAM (20 FPS): {current_len}/{SEQUENCE_LENGTH}"
                status_color = (0, 0, 255)

                # Progress bar di bawah layar
                progress_w = int((current_len / SEQUENCE_LENGTH) * (w - 40))
                cv2.rectangle(frame, (20, h - 35), (20 + progress_w, h - 20), (0, 0, 255), -1)

                if current_len == SEQUENCE_LENGTH:
                    # Simpan data .npy DAN video .mp4
                    next_idx = get_next_sample_index(current_class)
                    saved_base = save_sequence_data(current_class, next_idx, recorded_features, recorded_raw_frames, (w, h))

                    if burst_mode:
                        burst_count_session += 1
                        state = "BURST_REST"
                        burst_rest_start_time = time.time()
                    else:
                        state = "IDLE"
                        status_msg = f"✔ TERSIMPAN ({saved_base})! Tekan SPASI atau C"
                        status_color = (0, 255, 0)
            else:
                # Batalkan jika tangan hilang di tengah sekuens
                if burst_mode:
                    status_msg = "[BATAL] Tangan hilang! Posisikan tangan kembali..."
                    status_color = (0, 0, 255)
                    # Beri jeda sejenak lalu coba lagi
                    state = "BURST_REST"
                    burst_rest_start_time = time.time()
                else:
                    state = "IDLE"
                    recorded_features = []
                    recorded_raw_frames = []
                    status_msg = "[BATAL] Tangan tidak terdeteksi! Ulangi SPASI"
                    status_color = (0, 0, 255)

        elif state == "BURST_REST":
            rest_elapsed = time.time() - burst_rest_start_time
            rest_remaining = BURST_REST_DURATION - rest_elapsed
            if rest_remaining > 0:
                status_msg = f"AUTO-BURST: Berhasil {burst_count_session} sampel! Jeda istirahat ({rest_remaining:.1f}s)"
                status_color = (0, 255, 255)
                # Tampilkan visual istirahat/reset tangan
                cv2.rectangle(frame, (w // 2 - 200, h // 2 - 40), (w // 2 + 200, h // 2 + 40), (40, 40, 40), -1)
                cv2.putText(frame, f"ATUR TANGAN ({rest_remaining:.1f}s)", (w // 2 - 170, h // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                # Masuk langsung ke perekaman sampel berikutnya
                if hand_present:
                    state = "RECORDING"
                    recorded_features = []
                    recorded_raw_frames = []
                    last_frame_recorded_time = 0
                else:
                    # Tunggu hingga tangan masuk kembali ke layar
                    status_msg = "AUTO-BURST: Menunggu tangan masuk ke layar..."
                    status_color = (0, 165, 255)

        # --- Panel Informasi Overlay GUI ---
        # Kotak Header Status
        header_h = 125 if burst_mode else 115
        cv2.rectangle(frame, (10, 10), (510, header_h), (25, 25, 25), -1)
        cv2.putText(frame, f"Kelas [{current_class_idx + 1}/9]: {current_class} (Npy:{npy_cnt} | Vid:{mp4_cnt})", 
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_msg}", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2)

        if burst_mode:
            cv2.putText(frame, f"[AUTO-BURST ON: {burst_count_session} sesi] Tekan [C] atau [SPASI] utk STOP", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1)
            cv2.putText(frame, f"FPS: {int(fps)} | [D]: Hapus | [1-9]: Kelas | [M]: Mirror", (20, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        else:
            cv2.putText(frame, f"FPS: {int(fps)} | [SPASI]: 1x Rekam | [C]: Auto-Burst | [D]: Hapus", (20, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Daftar Mini Kelas di sisi kanan atas
        menu_y = 25
        for i, c_name in enumerate(GESTURE_CLASSES):
            c_color = (0, 255, 128) if i == current_class_idx else (160, 160, 160)
            n_c, _ = count_samples(c_name)
            cv2.putText(frame, f"{i+1}.{c_name} ({n_c})", (w - 175, menu_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_color, 1)
            menu_y += 18

        cv2.imshow("Data Collector - Tugas Akhir Hilmy", frame)

        # --- Penanganan Input Tombol Keyboard ---
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Tombol SPASI
            if burst_mode:
                # Matikan mode burst jika sedang aktif
                burst_mode = False
                state = "IDLE"
                status_msg = f"Auto-Burst dihentikan ({burst_count_session} sampel diambil)."
                status_color = (255, 255, 255)
            elif state == "IDLE":
                state = "COUNTDOWN"
                countdown_start_time = time.time()

        elif key == ord('c') or key == ord('C'):  # Tombol 'C' (Toggle Auto-Burst)
            if burst_mode:
                burst_mode = False
                state = "IDLE"
                status_msg = f"Auto-Burst dihentikan ({burst_count_session} sampel diambil)."
                status_color = (255, 255, 255)
            else:
                burst_mode = True
                burst_count_session = 0
                state = "COUNTDOWN"
                countdown_start_time = time.time()

        elif ord('1') <= key <= ord('9'):
            # Langsung pilih kelas dengan tombol angka 1-9
            burst_mode = False
            current_class_idx = key - ord('1')
            state = "IDLE"
            status_msg = f"Beralih ke kelas {GESTURE_CLASSES[current_class_idx]}"
            status_color = (255, 255, 255)

        elif key == 9 or key == ord('n') or key == ord('N'):  # Tombol TAB atau 'N'
            burst_mode = False
            current_class_idx = (current_class_idx + 1) % len(GESTURE_CLASSES)
            state = "IDLE"
            status_msg = f"Beralih ke kelas {GESTURE_CLASSES[current_class_idx]}"
            status_color = (255, 255, 255)

        elif key == ord('b') or key == ord('B'):
            burst_mode = False
            current_class_idx = (current_class_idx - 1) % len(GESTURE_CLASSES)
            state = "IDLE"

        elif key == 8 or key == ord('d') or key == ord('D'):  # BACKSPACE atau tombol 'D'
            deleted_info = delete_last_sample(current_class)
            if deleted_info:
                print(f"🗑️ [{current_class}] Berhasil dihapus: {deleted_info}")
                status_msg = f"🗑️ Terhapus: {deleted_info}"
                status_color = (0, 165, 255)
            else:
                status_msg = "Belum ada sampel untuk dihapus"
                status_color = (0, 0, 255)

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
    print("\nProgram selesai. Sampel tersimpan rapi di 'dataset/' dan 'raw_videos/'.")

if __name__ == "__main__":
    main()
