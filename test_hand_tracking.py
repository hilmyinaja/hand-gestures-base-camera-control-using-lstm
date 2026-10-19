import cv2
import time
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 Titik Landmark Sendi Tangan (Standard MediaPipe Hand Topology)
HAND_CONNECTIONS = [
    # Jempol (Thumb)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Telunjuk (Index finger)
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Jari Tengah (Middle finger)
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Jari Manis (Ring finger)
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Kelingking & Telapak (Pinky & Palm)
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def ensure_model_exists(model_path="hand_landmarker.task"):
    """Mengunduh file model MediaPipe jika belum ada di lokal."""
    if not os.path.exists(model_path):
        print(f"[Info] Mengunduh model {model_path} dari Google MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("[Info] Model berhasil diunduh!")

def main():
    print("==================================================")
    print("   MediaPipe Hands Test - Tugas Akhir Hilmy       ")
    print("==================================================")

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    ensure_model_exists(model_path)

    # Inisialisasi HandLandmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Buka Webcam Laptop (Default index: 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Peringatan] Kamera index 0 tidak dapat dibuka. Mencoba index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[Error] Tidak dapat mengakses webcam! Pastikan webcam tidak sedang dipakai aplikasi lain.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mirror_view = True
    print("\nWebcam aktif! Arahkan tanganmu ke kamera.")
    print("Tekan 'm' untuk toggle Mirror (Cermin) ON/OFF.")
    print("Tekan 'q' atau 'ESC' untuk menutup jendela.")

    start_time = time.time()
    prev_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Gagal membaca frame dari webcam.")
            break

        # Efek cermin jika mirror_view aktif
        if mirror_view:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Konversi ke RGB dan MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Hitung timestamp dalam milidetik (diperlukan untuk mode VIDEO)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Inferensi deteksi landmark tangan
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Hitung FPS
        now = time.time()
        fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else 0
        prev_frame_time = now

        hand_count = 0
        if detection_result and detection_result.hand_landmarks:
            hand_count = len(detection_result.hand_landmarks)

            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Deteksi label tangan (Kiri/Kanan)
                hand_label = "Tangan"
                if detection_result.handedness and i < len(detection_result.handedness):
                    raw_label = detection_result.handedness[i][0].category_name
                    # Saat frame di-mirror (cermin), orientasi visual terbalik dari perspektif kamera,
                    # sehingga 'Left' dari model sebenarnya adalah tangan 'Kanan' pengguna, dan sebaliknya.
                    if mirror_view:
                        hand_label = "Kanan" if raw_label == "Left" else "Kiri"
                    else:
                        hand_label = "Kanan" if raw_label == "Right" else "Kiri"

                # Kumpulkan titik koordinat pixel (x, y)
                coords = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    coords.append((cx, cy))

                # Gambar garis sendi (Bone Connections)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 128), 2)

                # Gambar titik sendi (Landmarks)
                for idx, pt in enumerate(coords):
                    # Titik 0 (Wrist / Pergelangan) dibuat lebih besar sebagai acuan normalisasi
                    if idx == 0:
                        cv2.circle(frame, pt, 7, (0, 0, 255), -1)  # Merah untuk Wrist
                        cv2.putText(frame, f"Tangan {hand_label} (Wrist)", (pt[0] - 50, pt[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    else:
                        cv2.circle(frame, pt, 4, (255, 100, 0), -1)

        # Overlay Informasi Sistem
        cv2.rectangle(frame, (10, 10), (370, 95), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {int(fps)} | Tangan: {hand_count}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        mode_text = "Mirror: ON (Alami)" if mirror_view else "Mirror: OFF (Asli)"
        cv2.putText(frame, f"{mode_text} [Tekan 'm']", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("MediaPipe Hands Test - Tugas Akhir Hilmy", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m') or key == ord('M'):
            mirror_view = not mirror_view
            print(f"[Info] Mode cermin diubah: {'ON' if mirror_view else 'OFF'}")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        detector.close()
    except:
        pass
    print("Selesai.")

if __name__ == "__main__":
    main()
