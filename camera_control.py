# command auto attach kamera
# usbipd attach --wsl --busid 1-1 --auto-attach
# (ganti busid sesuai yang ada di usbipd list)
   
import gphoto2 as gp
import time
import os

class CameraControl:
    def __init__(self):
        self.camera = gp.Camera()
        self.context = gp.Context()
        
        # Buat folder khusus secara otomatis
        self.photo_dir = "hasil_jepretan"
        self.video_dir = "hasil_rekaman"
        os.makedirs(self.photo_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def connect(self):
        """Connects to the camera."""
        try:
            self.camera.init()
            print("Successfully connected to the camera.")
        except gp.GPhoto2Error as e:
            print(f"Failed to connect to camera: {e}")
            raise

    def disconnect(self):
        """Disconnects from the camera."""
        self.camera.exit()
        print("Camera disconnected.")

    def _get_config(self):
        return self.camera.get_config()

    def _set_config(self, config):
        self.camera.set_config(config)

    def _get_config_value(self, config_name):
        """Helper to get a configuration value."""
        try:
            config = self._get_config()
            widget = config.get_child_by_name(config_name)
            return widget.get_value()
        except gp.GPhoto2Error:
            print(f"Error: Could not get config '{config_name}'.")
            return None

    def _set_config_value(self, config_name, value):
        """Helper to set a configuration value."""
        try:
            config = self._get_config()
            widget = config.get_child_by_name(config_name)
            widget.set_value(value)
            self._set_config(config)
            print(f"Set {config_name} to {value}")
        except gp.GPhoto2Error as e:
            print(f"Error setting {config_name} to {value}: {e}")

    # --- Properties Control ---

    def set_iso(self, iso_value):
        """Sets the ISO value."""
        self._set_config_value('iso', str(iso_value))

    def get_iso(self):
        """Gets the current ISO value."""
        return self._get_config_value('iso')

    def set_aperture(self, aperture_value):
        """Sets the aperture (f-number)."""
        self._set_config_value('f-number', str(aperture_value))

    def get_aperture(self):
        """Gets the current aperture."""
        return self._get_config_value('f-number')

    def set_shutter_speed(self, speed):
        """Sets the shutter speed."""
        self._set_config_value('shutterspeed', str(speed))

    def get_shutter_speed(self):
        """Gets the current shutter speed."""
        return self._get_config_value('shutterspeed')

    # --- Capture Control ---

    def reconnect(self):
        """Attempts to reconnect to the camera after a drop."""
        print("\n[Sistem] Koneksi terputus. Mencoba menyambung kembali dalam 4 detik...")
        time.sleep(4)
        try:
            self.camera.exit()
        except:
            pass
        self.camera = gp.Camera()
        self.context = gp.Context()
        try:
            self.connect()
            print("[Sistem] Berhasil menyambung kembali!")
        except Exception as e:
            print(f"[Sistem] Gagal menyambung kembali: {e}\nPastikan kamu sudah menggunakan flag --auto-attach di Windows.")

    def capture_photo(self, download=True, target_path=None):
        """Captures a photo using trigger_capture to avoid USB reset on Sony."""
        if target_path is None:
            target_path = self.photo_dir
            
        try:
            print("Memotret foto...")
            # Menggunakan trigger_capture agar mesin kamera tidak 'kaget' dan nge-reset USB
            self.camera.trigger_capture()
            
            if download:
                print("Menunggu file diproses oleh memori kamera...")
                # Kita tunggu secara pasif hingga kamera memberi sinyal bahwa foto selesai dibuat
                timeout = 5000  # 5 detik maksimal tunggu
                while timeout > 0:
                    event_type, event_data = self.camera.wait_for_event(1000)
                    
                    if event_type == gp.GP_EVENT_FILE_ADDED:
                        file_path = event_data
                        print(f"Foto terdeteksi di {file_path.folder}/{file_path.name}")
                        
                        target_file = os.path.join(target_path, file_path.name)
                        print(f"Mengunduh foto ke laptop ({target_file})...")
                        camera_file = self.camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
                        camera_file.save(target_file)
                        print(f"Unduhan selesai! Tersimpan di folder: {target_path}")
                        break
                    
                    timeout -= 1000
                    
                if timeout <= 0:
                    print("Waktu habis menunggu foto tersimpan (Apakah SD Card penuh/tidak ada?).")
            else:
                print("Foto berhasil dijepret (Hanya disimpan di SD Card kamera).")
                
        except gp.GPhoto2Error as e:
            print(f"Kamera mendadak terputus setelah memotret: {e}")
            self.reconnect()

    def start_recording(self):
        """Starts video recording. Note: Implementation depends heavily on camera model."""
        try:
            # Many cameras use the 'movie' config to toggle recording
            self._set_config_value('movie', 1)
            print("Video recording started.")
        except gp.GPhoto2Error:
            print("Failed to start recording. This feature may not be supported or requires a different config name.")

    def stop_recording(self):
        """Stops video recording."""
        try:
            self._set_config_value('movie', 0)
            print("Video recording stopped.")
        except gp.GPhoto2Error:
            print("Failed to stop recording.")

    # --- Lens Control (Power Zoom) ---
    
    def zoom(self, direction):
        """
        Controls the power zoom for Sony ILCE cameras.
        direction: 'in' or 'out'
        """
        try:
            if direction == 'in':
                # Kode rahasia Sony untuk tombol Zoom In (Tele) adalah d2c1
                self._set_config_value('d2c1', '2') # '2' = Tekan Tombol (String)
                time.sleep(0.5)                     # Tahan selama setengah detik
                self._set_config_value('d2c1', '1') # '1' = Lepas Tombol (String)
                print("Berhasil menekan tuas Zoom IN.")
                
            elif direction == 'out':
                # Kode rahasia Sony untuk tombol Zoom Out (Wide) adalah d2c2
                self._set_config_value('d2c2', '2') # '2' = Tekan Tombol (String)
                time.sleep(0.5)                     # Tahan selama setengah detik
                self._set_config_value('d2c2', '1') # '1' = Lepas Tombol (String)
                print("Berhasil menekan tuas Zoom OUT.")
                
            else:
                print("Arah zoom tidak valid. Gunakan 'in' atau 'out'.")

        except gp.GPhoto2Error as e:
            print(f"Error saat mengontrol zoom (Pastikan lensa Power Zoom terpasang): {e}")


if __name__ == "__main__":
    cam = CameraControl()
    try:
        cam.connect()
        
        while True:
            print("\n=== Menu Kontrol Kamera ===")
            print("1. Cek Status (ISO, Aperture, Shutter Speed)")
            print("2. Atur ISO")
            print("3. Atur Aperture")
            print("4. Atur Shutter Speed")
            print("5. Ambil Foto (Capture)")
            print("6. Mulai Rekam Video")
            print("7. Berhenti Rekam Video")
            print("8. Zoom In (Power Zoom)")
            print("9. Zoom Out (Power Zoom)")
            print("0. Keluar")
            
            pilihan = input("\nMasukkan pilihanmu (0-9): ")
            print("-" * 30)
            
            if pilihan == '0':
                print("Keluar dari program...")
                break
            elif pilihan == '1':
                print(f"ISO saat ini: {cam.get_iso()}")
                print(f"Aperture saat ini: {cam.get_aperture()}")
                print(f"Shutter Speed saat ini: {cam.get_shutter_speed()}")
            elif pilihan == '2':
                val = input("Masukkan nilai ISO (contoh: 100, 800, 1600): ")
                cam.set_iso(val)
            elif pilihan == '3':
                val = input("Masukkan nilai Aperture/F-Number (contoh: 2.8, 5.6): ")
                cam.set_aperture(val)
            elif pilihan == '4':
                val = input("Masukkan nilai Shutter Speed (contoh: 1/100, 1/250): ")
                cam.set_shutter_speed(val)
            elif pilihan == '5':
                cam.capture_photo(download=True)
            elif pilihan == '6':
                cam.start_recording()
            elif pilihan == '7':
                cam.stop_recording()
            elif pilihan == '8':
                # cam.zoom('in')
                print("Fitur zoom dinonaktifkan sementara (Bisa diaktifkan kembali jika ganti kamera).")
            elif pilihan == '9':
                # cam.zoom('out')
                print("Fitur zoom dinonaktifkan sementara (Bisa diaktifkan kembali jika ganti kamera).")
            else:
                print("Pilihan tidak valid, silakan coba lagi.")
                
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
    finally:
        cam.disconnect()

