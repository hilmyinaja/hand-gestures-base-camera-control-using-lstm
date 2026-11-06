"""
Pelatihan Model Stacked LSTM untuk Pengenalan Gestur Kontrol Kamera
Dengan Sistem Versioning Eksperimen (Experiment Tracking & Model Versioning)
Berdasarkan Bab 3 Metodologi Proposal Tugas Akhir
Hilmy Abid Syafi Abiyyu (NRP 5024231029) - Teknik Komputer FTEIC ITS
"""

import os
import sys
import glob
import json
import csv
import shutil
import tempfile
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Pastikan cache matplotlib tidak menimbulkan izin akses error di Windows
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Konfigurasi Sesuai Bab 3 Proposal ---
DATASET_DIR = "dataset"
RUNS_DIR = "runs"

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

SEQUENCE_LENGTH = 20  # 20 frame per sekuens
NUM_FEATURES = 63     # 21 koordinat 3D (x, y, z)
NUM_CLASSES = len(GESTURE_CLASSES)

BATCH_SIZE = 32       # Sesuai Subbab 3.3.2 Poin 3
MAX_EPOCHS = 100      # Sesuai Subbab 3.3.2 Poin 4
LEARNING_RATE = 0.001 # Sesuai Subbab 3.3.2 Poin 1
PATIENCE = 15         # Sesuai Subbab 3.3.2 Poin 5

# Lokasi File Root (Untuk Akses Instan main_gesture_control.py)
ROOT_MODEL_KERAS = "model_gesture.keras"
ROOT_MODEL_TFLITE = "model_gesture.tflite"
CLASSES_JSON_PATH = "gesture_classes.json"
ROOT_CONFUSION_MATRIX = "confusion_matrix.png"
ROOT_TRAINING_HISTORY = "training_history.png"

def setup_run_directory():
    """
    Membuat folder terisolasi untuk iterasi pelatihan saat ini (runs/run_XXX_YYYYMMDD_HHMM/).
    Mencegah penimpaan hasil eksperimen sebelumnya untuk keperluan Bab 4 Skripsi.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    existing_runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("run_")]
    
    nums = []
    for d in existing_runs:
        try:
            num = int(d.split("_")[1])
            nums.append(num)
        except (ValueError, IndexError):
            pass
            
    next_num = (max(nums) + 1) if nums else 1
    run_id = f"run_{next_num:03d}"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder_name = f"{run_id}_{timestamp_str}"
    run_folder_path = os.path.join(RUNS_DIR, run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)

    print(f"=================================================================")
    print(f"   EKSPERIMEN ITERASI PELATIHAN: [{run_id.upper()}]              ")
    print(f"   Folder Output : {run_folder_path}                             ")
    print(f"=================================================================\n")

    return run_id, run_folder_path

def load_dataset():
    """Memuat seluruh file sekuens .npy dari folder dataset/."""
    print("=================================================================")
    print("   1. PEMUATAN DATASET SEKUENS (.npy)                            ")
    print("=================================================================")

    X_list = []
    y_list = []

    for class_idx, class_name in enumerate(GESTURE_CLASSES):
        class_folder = os.path.join(DATASET_DIR, class_name)
        file_paths = sorted(glob.glob(os.path.join(class_folder, "*.npy")))
        print(f"  • Memuat kelas [{class_idx}] {class_name:<14}: {len(file_paths)} sampel")

        if len(file_paths) == 0:
            print(f"[Peringatan] Folder '{class_folder}' masih kosong!")

        for fp in file_paths:
            try:
                seq = np.load(fp)
                if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    X_list.append(seq)
                    y_list.append(class_idx)
                else:
                    print(f"[Peringatan] Dimensi tidak sesuai {seq.shape} di {fp}, diabaikan.")
            except Exception as e:
                print(f"[Error] Gagal membaca {fp}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\nTotal Sampel Berhasil Dimuat : {len(X)}")
    print(f"Bentuk Matriks Fitur (X)      : {X.shape}")
    print(f"Bentuk Vektor Label (y)       : {y.shape}")

    # Simpan mapping kelas ke file JSON
    with open(CLASSES_JSON_PATH, "w") as f:
        json.dump(GESTURE_CLASSES, f, indent=4)
    print(f"Mapping kelas tersimpan di    : {CLASSES_JSON_PATH}\n")

    return X, y

def split_dataset(X, y):
    """
    Membagi dataset sesuai Subbab 3.2.3:
    - 70% Data Latih (Training Set)
    - 15% Data Validasi (Validation Set)
    - 15% Data Uji (Testing Set)
    Menggunakan Stratified Random Sampling agar seimbang antar-kelas.
    """
    print("=================================================================")
    print("   2. PEMBAGIAN DATASET (STRATIFIED RANDOM SAMPLING)             ")
    print("=================================================================")

    # Pisahkan 15% untuk Test Set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    # Dari 85% sisanya, pisahkan 15% dari total awal untuk Validation Set
    val_ratio = 0.15 / (0.70 + 0.15)  # ~17.65% dari train_val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=42
    )

    # One-Hot Encoding untuk target klasifikasi
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    print(f"  • Data Latih    (Train 70%) : {len(X_train)} sampel ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Data Validasi (Val 15%)   : {len(X_val)} sampel ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  • Data Uji      (Test 15%)  : {len(X_test)} sampel ({len(X_test)/len(X)*100:.1f}%)\n")

    return (X_train, y_train_cat, y_train), (X_val, y_val_cat, y_val), (X_test, y_test_cat, y_test)

def build_stacked_lstm_model():
    """
    Membangun arsitektur Stacked LSTM sesuai Subbab 3.3.1:
    1. Input Layer: (20, 63)
    2. LSTM Layer 1: 64 hidden units (return_sequences=True, unroll=True, tanh, sigmoid)
    3. Dropout Layer 1: 0.2
    4. LSTM Layer 2: 64 hidden units (return_sequences=False, unroll=True, tanh, sigmoid)
    5. Dropout Layer 2: 0.2
    6. Fully Connected (Dense): 64 neuron (ReLU)
    7. Output Layer: NUM_CLASSES neuron (Softmax)
    """
    print("=================================================================")
    print("   3. MEMBANGUN ARSITEKTUR STACKED LSTM                          ")
    print("=================================================================")

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name="Input_Landmarks_Sequence"),
        LSTM(64, return_sequences=True, unroll=True, activation='tanh', recurrent_activation='sigmoid', name="LSTM_Layer_1"),
        Dropout(0.2, name="Dropout_1"),
        LSTM(64, return_sequences=False, unroll=True, activation='tanh', recurrent_activation='sigmoid', name="LSTM_Layer_2"),
        Dropout(0.2, name="Dropout_2"),
        Dense(64, activation='relu', name="Dense_Features"),
        Dense(NUM_CLASSES, activation='softmax', name="Output_Softmax_Probabilities")
    ], name="Stacked_LSTM_Gesture_Classifier")

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    print()
    return model

def train_model(model, train_data, val_data, run_folder_path):
    """
    Melatih model dengan konfigurasi hiperparameter sesuai Subbab 3.3.2:
    - Optimizer: Adam (lr = 0.001)
    - Loss: Categorical Cross-Entropy
    - Max Epoch: 100
    - Batch Size: 32 (atau adaptif jika sampel kecil)
    - Early Stopping (patience=15 pada val_loss)
    - Model Checkpoint (simpan bobot terbaik ke folder run)
    """
    print("=================================================================")
    print("   4. PROSES PELATIHAN MODEL (TRAINING & VALIDATION)             ")
    print("=================================================================")

    X_train, y_train_cat, _ = train_data
    X_val, y_val_cat, _ = val_data

    run_keras_path = os.path.join(run_folder_path, "model_gesture.keras")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=run_keras_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    effective_batch_size = min(BATCH_SIZE, len(X_train))

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        batch_size=effective_batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Sinkronisasi model Keras terbaik ke root
    shutil.copyfile(run_keras_path, ROOT_MODEL_KERAS)
    print(f"\n[OK] Model Keras terbaik tersimpan di:")
    print(f"     • Run Folder : {run_keras_path}")
    print(f"     • Root File  : {ROOT_MODEL_KERAS}")

    return history

def evaluate_and_save_artifacts(model, test_data, history, run_id, run_folder_path, total_samples):
    """
    Evaluasi performa pada data uji (testing set) sesuai Subbab 3.5.1:
    - Akurasi, Presisi, Recall, F1-Score
    - Matriks Konfusi (Confusion Matrix)
    - Grafik Kurva Pelatihan (Loss & Akurasi)
    - Ekspor laporan detail metrics.json & summary_report.txt ke runs/
    """
    print("\n=================================================================")
    print("   5. EVALUASI MODEL PADA DATA UJI (TESTING SET)                ")
    print("=================================================================")

    X_test, y_test_cat, y_test = test_data

    # Evaluasi kuantitatif
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  • Test Loss     : {test_loss:.4f}")
    print(f"  • Test Accuracy : {test_acc * 100:.2f}%\n")

    # Prediksi kelas
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    report_dict = classification_report(y_test, y_pred, target_names=GESTURE_CLASSES, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, target_names=GESTURE_CLASSES, digits=4, zero_division=0)
    print("Classification Report:")
    print(report_str)

    # Matriks Konfusi
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    # 1. Plot & Simpan Confusion Matrix
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({run_id.upper()} - Akurasi: {test_acc*100:.1f}%)", fontsize=13, pad=15)
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, GESTURE_CLASSES, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, GESTURE_CLASSES, fontsize=9)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, 'd'),
                     ha="center", va="center",
                     color=color, fontsize=10, fontweight='bold')

    plt.ylabel('Label Sebenarnya (True Label)', fontsize=11)
    plt.xlabel('Label Prediksi Model (Predicted Label)', fontsize=11)
    plt.tight_layout()

    run_cm_path = os.path.join(run_folder_path, "confusion_matrix.png")
    plt.savefig(run_cm_path, dpi=300)
    plt.savefig(ROOT_CONFUSION_MATRIX, dpi=300)
    plt.close()
    print(f"[OK] Gambar Confusion Matrix tersimpan di : {run_cm_path}")

    # 2. Plot & Simpan Training History
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='#ff7f0e', linewidth=2)
    plt.title(f'Kurva Akurasi Pelatihan ({run_id.upper()})', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Akurasi', fontsize=10)
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title(f'Kurva Nilai Kerugian / Loss ({run_id.upper()})', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    run_hist_path = os.path.join(run_folder_path, "training_history.png")
    plt.savefig(run_hist_path, dpi=300)
    plt.savefig(ROOT_TRAINING_HISTORY, dpi=300)
    plt.close()
    print(f"[OK] Gambar Kurva Pelatihan tersimpan di   : {run_hist_path}")

    # 3. Simpan Metrics JSON & Text Summary untuk Bab 4
    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    total_epochs = len(history.history['loss'])

    metrics_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_samples": total_samples,
        "epochs_trained": total_epochs,
        "best_epoch": best_epoch,
        "train_accuracy_final": float(history.history['accuracy'][-1]),
        "val_accuracy_final": float(history.history['val_accuracy'][-1]),
        "val_loss_best": float(np.min(history.history['val_loss'])),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "macro_avg_f1_score": float(report_dict['macro avg']['f1-score']),
        "weighted_avg_f1_score": float(report_dict['weighted avg']['f1-score']),
        "classification_report": report_dict
    }

    metrics_json_path = os.path.join(run_folder_path, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[OK] Metrik JSON tersimpan di             : {metrics_json_path}")

    report_txt_path = os.path.join(run_folder_path, "summary_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"=================================================================\n")
        f.write(f"   RINGKASAN EKSPERIMEN {run_id.upper()} (UNTUK BAB 4 SKRIPSI)   \n")
        f.write(f"=================================================================\n")
        f.write(f"Waktu Pelatihan      : {metrics_summary['timestamp']}\n")
        f.write(f"Total Sampel Dataset : {total_samples} sampel (9 kelas)\n")
        f.write(f"Epoch Dilatih        : {total_epochs} (Best Epoch: {best_epoch})\n")
        f.write(f"Akurasi Uji (Test)   : {test_acc * 100:.2f}%\n")
        f.write(f"Loss Uji (Test)      : {test_loss:.4f}\n")
        f.write(f"Macro F1-Score       : {metrics_summary['macro_avg_f1_score']:.4f}\n\n")
        f.write("Classification Report Lengkap:\n")
        f.write(report_str)
    print(f"[OK] Ringkasan teks Bab 4 tersimpan di    : {report_txt_path}")

    return metrics_summary

def convert_to_tflite(model, run_folder_path):
    """
    Konversi Model ke format TensorFlow Lite (.tflite) sesuai Subbab 3.3.3:
    Menggunakan Post-Training Quantization FP16 untuk memangkas ukuran
    dan mempercepat inferensi pada prosesor ARM Raspberry Pi.
    """
    print("\n=================================================================")
    print("   6. OPTIMASI & KONVERSI KE TENSORFLOW LITE (.tflite FP16)     ")
    print("=================================================================")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    run_tflite_path = os.path.join(run_folder_path, "model_gesture.tflite")
    with open(run_tflite_path, "wb") as f:
        f.write(tflite_model)

    # Sinkronisasi model TFLite ke root agar main_gesture_control.py langsung memakai versi terbaru
    shutil.copyfile(run_tflite_path, ROOT_MODEL_TFLITE)

    keras_size = os.path.getsize(ROOT_MODEL_KERAS) / 1024.0
    tflite_size = os.path.getsize(run_tflite_path) / 1024.0

    print(f"[OK] Model TFLite disimpan di        : {run_tflite_path} ({tflite_size:.1f} KB)")
    print(f"[OK] Model TFLite Root di-update ke  : {ROOT_MODEL_TFLITE}")
    print(f"[OK] Efisiensi Kompresi Model       : Memangkas memori sebesar {(1 - tflite_size/keras_size)*100:.1f}%\n")

    # Uji Coba Cepat TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=ROOT_MODEL_TFLITE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  • TFLite Input Shape  : {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  • TFLite Output Shape : {output_details[0]['shape']} ({output_details[0]['dtype']})")
    print("[OK] TFLite Interpreter berhasil diverifikasi dan siap jalan di Raspberry Pi!")

def update_experiment_history_log(metrics_summary, run_folder_path):
    """
    Menambahkan 1 baris rekapitulasi ke file CSV dan Markdown di folder runs/.
    Ini memberikan tabel evolusi akurasi yang siap disalin ke Bab 4 Skripsi!
    """
    csv_path = os.path.join(RUNS_DIR, "experiments_history.csv")
    md_path = os.path.join(RUNS_DIR, "experiments_history.md")

    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "Run_ID", "Timestamp", "Total_Samples", "Epochs_Trained",
        "Best_Epoch", "Train_Acc", "Val_Acc", "Test_Acc", "Test_Loss",
        "Macro_F1", "Weighted_F1", "Run_Folder"
    ]

    row_data = {
        "Run_ID": metrics_summary["run_id"],
        "Timestamp": metrics_summary["timestamp"],
        "Total_Samples": metrics_summary["total_dataset_samples"],
        "Epochs_Trained": metrics_summary["epochs_trained"],
        "Best_Epoch": metrics_summary["best_epoch"],
        "Train_Acc": f"{metrics_summary['train_accuracy_final']*100:.2f}%",
        "Val_Acc": f"{metrics_summary['val_accuracy_final']*100:.2f}%",
        "Test_Acc": f"{metrics_summary['test_accuracy']*100:.2f}%",
        "Test_Loss": f"{metrics_summary['test_loss']:.4f}",
        "Macro_F1": f"{metrics_summary['macro_avg_f1_score']:.4f}",
        "Weighted_F1": f"{metrics_summary['weighted_avg_f1_score']:.4f}",
        "Run_Folder": os.path.basename(run_folder_path)
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Tulis ulang tabel Markdown ringkasan untuk mudah dibaca di GitHub/VS Code
    all_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tabel Rekapitulasi Riwayat Pelatihan Model (Bab 4 Skripsi)\n\n")
        f.write("Tabel ini mencatat seluruh eksperimen pelatihan model Stacked LSTM secara otomatis:\n\n")
        f.write("| Run ID | Waktu | Total Sampel | Epoch | Best Epoch | Train Acc | Val Acc | Test Acc | Test Loss | Macro F1 |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in all_rows:
            f.write(f"| `{r['Run_ID']}` | {r['Timestamp']} | {r['Total_Samples']} | {r['Epochs_Trained']} | {r['Best_Epoch']} | {r['Train_Acc']} | {r['Val_Acc']} | **{r['Test_Acc']}** | {r['Test_Loss']} | {r['Macro_F1']} |\n")

    print(f"\n[OK] Log Riwayat Eksperimen berhasil diperbarui di:")
    print(f"     • CSV : {csv_path}")
    print(f"     • MD  : {md_path}")

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"[Error] Folder '{DATASET_DIR}' tidak ditemukan!")
        return

    # 0. Siapkan Direktori Run Eksperimen
    run_id, run_folder_path = setup_run_directory()

    # 1. Muat Dataset
    X, y = load_dataset()
    if len(X) == 0:
        print("[Error] Tidak ada sampel yang berhasil dimuat. Rekam data terlebih dahulu!")
        return

    # 2. Split Dataset
    train_data, val_data, test_data = split_dataset(X, y)

    # 3. Bangun Model
    model = build_stacked_lstm_model()

    # 4. Latih Model
    history = train_model(model, train_data, val_data, run_folder_path)

    # 5. Evaluasi & Simpan Seluruh Artefak
    metrics_summary = evaluate_and_save_artifacts(model, test_data, history, run_id, run_folder_path, len(X))

    # 6. Konversi ke TFLite
    convert_to_tflite(model, run_folder_path)

    # 7. Update Log Rekapitulasi Eksperimen
    update_experiment_history_log(metrics_summary, run_folder_path)

    print("\n=================================================================")
    print(f"   🎉 EKSPERIMEN {run_id.upper()} SELESAI DENGAN SUKSES!           ")
    print(f"   Semua artefak tersimpan di: {run_folder_path}                 ")
    print("=================================================================")

if __name__ == "__main__":
    main()
