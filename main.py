import cv2
import numpy as np
import customtkinter as ctk
import time
import face_recognition

from ultralytics import YOLO
from PIL import Image
from customtkinter import CTkImage, CTkTabview
from face_recog import FaceRecognition, face_confidence

class VideoProcessor:
    def __init__(self):
        # YOLO modellerini yükle
        self.detection_model = YOLO('yolov8n.pt')
        self.segmentation_model = YOLO('yolov8n-seg.pt')  # Segmentasyon modeli
        self.cap = cv2.VideoCapture(0)  # Kamera kullanımı
        
        # Kamera ayarları
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Buffer boyutunu artır
        
        # Frame buffer
        self.frame_buffer = []
        self.buffer_size = 3
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        
        # Yüz tanıma sistemini başlat
        self.face_recognition = FaceRecognition()
        
        # Görüntüleme modu
        self.view_mode = "detection"  # "detection" veya "segmentation"
        
        # Efekt durumları ve parametreleri
        self.person_effects = {
            "denoise": {"active": False, "strength": 10},
            "night_vision": {"active": False, "brightness": 1.3},
            "enhance": {"active": False, "contrast": 1.2, "brightness": 1.1},
            "sharpen": {"active": False, "strength": 1.5},
            "hdr": {"active": False, "exposure": 1.2}
        }
        
        self.background_effects = {
            "denoise": {"active": False, "strength": 10},
            "night_vision": {"active": False, "brightness": 1.3},
            "enhance": {"active": False, "contrast": 1.2, "brightness": 1.1},
            "sharpen": {"active": False, "strength": 1.5},
            "hdr": {"active": False, "exposure": 1.2}
        }
        
        # Video kayıt değişkenleri
        self.is_recording = False
        self.video_writer = None
        self.record_start_time = None

    def apply_effects(self, frame, effect_type="person"):
        effects = self.person_effects if effect_type == "person" else self.background_effects
        
        if effects["denoise"]["active"]:
            # Gürültü azaltma işlemini optimize et
            strength = effects["denoise"]["strength"]
            # Görüntüyü küçült
            small_frame = cv2.resize(frame, (320, 240))
            # Gürültü azaltma uygula
            denoised = cv2.fastNlMeansDenoisingColored(small_frame, None, 
                                                     h=strength,
                                                     hColor=strength,
                                                     templateWindowSize=7,
                                                     searchWindowSize=21)
            # Görüntüyü orijinal boyuta getir
            frame = cv2.resize(denoised, (frame.shape[1], frame.shape[0]))
            
        if effects["night_vision"]["active"]:
            # Gece görüşü efekti
            brightness = effects["night_vision"]["brightness"]
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        if effects["enhance"]["active"]:
            # Görüntü iyileştirme
            contrast = effects["enhance"]["contrast"]
            brightness = effects["enhance"]["brightness"]
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness*50)
            
        if effects["sharpen"]["active"]:
            # Keskinleştirme
            strength = effects["sharpen"]["strength"]
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * strength
            frame = cv2.filter2D(frame, -1, kernel)
            
        if effects["hdr"]["active"]:
            # HDR efekti
            exposure = effects["hdr"]["exposure"]
            # Ton eşleme
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl,a,b))
            frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            frame = cv2.convertScaleAbs(frame, alpha=exposure, beta=0)
            
        return frame
        
    def start_recording(self):
        if not self.is_recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.mp4"
            
            # Video writer ayarları
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (640, 480)  # Sabit boyut
            
            try:
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
                if not self.video_writer.isOpened():
                    raise Exception("Video writer açılamadı")
                
                self.is_recording = True
                self.record_start_time = time.time()
                return True
            except Exception as e:
                print(f"Video kayıt hatası: {str(e)}")
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                return False
        return False
    
    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.release()
                self.video_writer = None
                self.is_recording = False
                return True
            except Exception as e:
                print(f"Video kayıt kapatma hatası: {str(e)}")
                return False
        return False
        
    def process_frame(self):
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Frame'ler arası süre kontrolü
        if elapsed < self.frame_interval:
            # Buffer'dan frame al
            if self.frame_buffer:
                frame = self.frame_buffer.pop(0)
                return frame
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)
        
        # Frame'i buffer'a ekle
        if len(self.frame_buffer) < self.buffer_size:
            self.frame_buffer.append(frame.copy())
        
        # Görüntüleme moduna göre model seç
        if self.view_mode == "detection":
            results = self.detection_model(frame, verbose=False)
        else:
            results = self.segmentation_model(frame, verbose=False)
        
        # Kişi tespiti için maske oluştur
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if self.view_mode == "detection":
            # Tespit edilen nesneleri çiz ve maske oluştur
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    
                    # Güven skoru 0.5'ten büyükse çiz
                    if conf > 0.5:
                        # Kişi tespit edildiyse maskeyi güncelle
                        if int(cls) == 0:  # class 0: person
                            mask[int(y1):int(y2), int(x1):int(x2)] = 255
                        
                        # Bbox çiz
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        
                        # Sınıf adı ve güven skorunu yaz
                        label = f"{result.names[int(cls)]} {conf:.2f}"
                        
                        # Metin arka planı
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1) - text_height - 10),
                                    (int(x1) + text_width + 10, int(y1)),
                                    (0, 255, 0), -1)
                        
                        # Metin
                        cv2.putText(frame, label, 
                                  (int(x1) + 5, int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 0), 2)
        else:  # segmentation mode
            # Segmentasyon maskelerini uygula
            for result in results:
                if result.masks is not None:
                    for mask_data in result.masks.data:
                        # Maskeyi frame boyutuna getir
                        mask_resized = cv2.resize(mask_data.cpu().numpy(), 
                                                (frame.shape[1], frame.shape[0]))
                        # Maskeyi binary formata dönüştür
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        # Maskeyi birleştir
                        mask = cv2.bitwise_or(mask, mask_binary)
            
            # Segmentasyon maskesini renklendir
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            # Orijinal görüntü ile maskeyi birleştir
            frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
            
            # Sınıf isimlerini ve güven skorlarını göster
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        cls = box.cls[0]
                        
                        if conf > 0.5:
                            # Sınıf adı ve güven skorunu yaz
                            label = f"{result.names[int(cls)]} {conf:.2f}"
                            
                            # Metin arka planı
                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, 
                                        (int(x1), int(y1) - text_height - 10),
                                        (int(x1) + text_width + 10, int(y1)),
                                        (0, 255, 0), -1)
                            
                            # Metin
                            cv2.putText(frame, label, 
                                      (int(x1) + 5, int(y1) - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 0, 0), 2)
        
        # Kişi ve arka plan için efektleri uygula
        person_frame = self.apply_effects(frame.copy(), "person")
        background_frame = self.apply_effects(frame.copy(), "background")
        
        # Maskeyi 3 kanala genişlet
        mask_3d = cv2.merge([mask, mask, mask])
        
        # Efektleri birleştir
        frame = np.where(mask_3d == 255, person_frame, background_frame)
        
        # Yüz tanıma işlemini uygula
        if self.face_recognition.process_current_frame:
            # Küçük karede yüz konumlarını ve kodlamalarını bul
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            self.face_recognition.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_recognition.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_recognition.face_locations)
            
            self.face_recognition.face_names = []
            for face_encoding in self.face_recognition.face_encodings:
                matches = face_recognition.compare_faces(self.face_recognition.known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 'unknown'
                
                face_distances = face_recognition.face_distance(self.face_recognition.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.face_recognition.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])
                
                self.face_recognition.face_names.append(f'{name} {confidence}')
        
        self.face_recognition.process_current_frame = not self.face_recognition.process_current_frame
        
        # Yüzlerin etrafına modern kutu çiz ve isimleri yaz
        for (top, right, bottom, left), name in zip(self.face_recognition.face_locations, self.face_recognition.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Modern yüz çerçevesi
            thickness = 2
            # Dış çerçeve
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), thickness + 2)
            # İç çerçeve
            cv2.rectangle(frame, (left + thickness, top + thickness), 
                        (right - thickness, bottom - thickness), (255, 255, 255), thickness)
            
            # Modern isim etiketi
            padding = 10
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            # Yarı saydam arka plan
            overlay = frame.copy()
            cv2.rectangle(overlay, (left, bottom + 5), 
                        (left + text_size[0] + padding * 2, bottom + text_size[1] + padding * 2),
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            # İsim ve güven skoru
            cv2.putText(frame, name, (left + padding, bottom + text_size[1] + padding),
                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Kayıt yapılıyorsa frame'i kaydet
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
        
        self.last_frame_time = current_time
        return frame
        
    def release(self):
        if self.is_recording:
            self.stop_recording()
        self.cap.release()

class ModernGUI:
    def __init__(self, video_processor):
        self.root = ctk.CTk()
        self.root.title("Nesne Tespiti ve Efektler")
        self.root.geometry("1200x800")  # Pencere boyutunu büyüt
        
        # Tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.video_processor = video_processor
        self.video_processor.root = self.root  # root referansını video processor'a ekle
        
        # Ana frame'ler
        self.video_frame = ctk.CTkFrame(self.root, fg_color="gray20", corner_radius=10)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        
        self.control_frame = ctk.CTkFrame(self.root, width=400)  # Kontrol paneli genişliği
        self.control_frame.pack(side="right", fill="y", padx=20, pady=20)
        
        # Video görüntüleme alanı
        self.video_label = ctk.CTkLabel(self.video_frame, text="", fg_color="black", corner_radius=5)
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Kontrol paneli başlığı
        title_label = ctk.CTkLabel(self.control_frame, 
                                 text="Efekt Kontrol Paneli",
                                 font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=20)
        
        # Görüntüleme modu seçimi
        mode_frame = ctk.CTkFrame(self.control_frame)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(mode_frame, 
                                text="Görüntüleme Modu:",
                                font=ctk.CTkFont(size=16))
        mode_label.pack(side="left", padx=10)
        
        self.mode_var = ctk.StringVar(value="detection")
        
        detection_radio = ctk.CTkRadioButton(mode_frame,
                                           text="Nesne Tespiti",
                                           variable=self.mode_var,
                                           value="detection",
                                           command=self.update_view_mode,
                                           font=ctk.CTkFont(size=14))
        detection_radio.pack(side="left", padx=10)
        
        segmentation_radio = ctk.CTkRadioButton(mode_frame,
                                              text="Segmentasyon",
                                              variable=self.mode_var,
                                              value="segmentation",
                                              command=self.update_view_mode,
                                              font=ctk.CTkFont(size=14))
        segmentation_radio.pack(side="left", padx=10)
        
        # Tab view oluştur
        self.tabview = CTkTabview(self.control_frame, width=350)  # Tab genişliği
        self.tabview.pack(fill="x", padx=20, pady=10)
        
        # Efektler ve parametreler sekmeleri
        self.tabview.add("Kişi Efektleri")
        self.tabview.add("Arka Plan Efektleri")
        self.tabview.add("Parametreler")
        
        # Efekt seçenekleri
        effects = [
            ("Gürültü Azaltma (fastNlMeansDenoisingColored)", "denoise"),
            ("Gece Görüşü (convertScaleAbs)", "night_vision"),
            ("Görüntü İyileştirme (convertScaleAbs)", "enhance"),
            ("Keskinleştirme (filter2D)", "sharpen"),
            ("HDR (CLAHE)", "hdr")
        ]
        
        # Kişi efektleri checkbox'ları
        self.person_effect_vars = {}
        for text, value in effects:
            var = ctk.BooleanVar(value=False)
            self.person_effect_vars[value] = var
            
            frame = ctk.CTkFrame(self.tabview.tab("Kişi Efektleri"))
            frame.pack(fill="x", pady=5, padx=10)
            
            checkbox = ctk.CTkCheckBox(
                frame,
                text=text,
                variable=var,
                command=lambda v=value: self.update_effect("person", v),
                font=ctk.CTkFont(size=14),
                width=300
            )
            checkbox.pack(side="left", padx=5)
        
        # Arka plan efektleri checkbox'ları
        self.background_effect_vars = {}
        for text, value in effects:
            var = ctk.BooleanVar(value=False)
            self.background_effect_vars[value] = var
            
            frame = ctk.CTkFrame(self.tabview.tab("Arka Plan Efektleri"))
            frame.pack(fill="x", pady=5, padx=10)
            
            checkbox = ctk.CTkCheckBox(
                frame,
                text=text,
                variable=var,
                command=lambda v=value: self.update_effect("background", v),
                font=ctk.CTkFont(size=14),
                width=300
            )
            checkbox.pack(side="left", padx=5)
        
        # Parametre slider'ları
        self.create_parameter_sliders()
        
        # Durum çubuğu
        self.status_label = ctk.CTkLabel(
            self.control_frame,
            text="Hazır",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(side="bottom", pady=20)
        
        self.update_video()
    
    def create_parameter_sliders(self):
        # Parametre slider'larını oluştur
        params = {
            "denoise": [("Güç", "strength", 1, 20, 10)],
            "night_vision": [("Parlaklık", "brightness", 0.5, 2.0, 1.3)],
            "enhance": [
                ("Kontrast", "contrast", 0.5, 2.0, 1.2),
                ("Parlaklık", "brightness", 0.5, 2.0, 1.1)
            ],
            "sharpen": [("Güç", "strength", 0.5, 3.0, 1.5)],
            "hdr": [("Pozlama", "exposure", 0.5, 2.0, 1.2)]
        }
        
        # Önce tüm widget'ları temizle
        for widget in self.tabview.tab("Parametreler").winfo_children():
            widget.destroy()
        
        # Aktif efektlerin parametrelerini göster
        active_effects = []
        if self.tabview.get() == "Kişi Efektleri":
            active_effects = [effect for effect, var in self.person_effect_vars.items() if var.get()]
            effects_dict = self.video_processor.person_effects
        else:
            active_effects = [effect for effect, var in self.background_effect_vars.items() if var.get()]
            effects_dict = self.video_processor.background_effects
        
        for effect in active_effects:
            # Efekt başlığı
            title = ctk.CTkLabel(
                self.tabview.tab("Parametreler"),
                text=effect.upper(),
                font=ctk.CTkFont(size=16, weight="bold")
            )
            title.pack(pady=(15, 5))
            
            for label, param, min_val, max_val, default_val in params[effect]:
                frame = ctk.CTkFrame(self.tabview.tab("Parametreler"))
                frame.pack(fill="x", pady=5, padx=10)
                
                label = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=14))
                label.pack(side="left", padx=5)
                
                # Değer göstergesi için label
                value_label = ctk.CTkLabel(frame, text=str(default_val), font=ctk.CTkFont(size=14))
                value_label.pack(side="right", padx=5)
                
                slider = ctk.CTkSlider(
                    frame,
                    from_=min_val,
                    to=max_val,
                    number_of_steps=int((max_val - min_val) * 10),
                    command=lambda v, l=value_label, p=param, e=effect: self.update_parameter_with_label(e, p, v, l)
                )
                slider.set(effects_dict[effect][param])
                slider.pack(side="right", fill="x", expand=True, padx=5)
    
    def update_parameter_with_label(self, effect, param, value, label):
        if self.tabview.get() == "Kişi Efektleri":
            self.video_processor.person_effects[effect][param] = value
        else:
            self.video_processor.background_effects[effect][param] = value
        label.configure(text=f"{value:.2f}")
    
    def update_effect(self, effect_type, effect_name):
        if effect_type == "person":
            self.video_processor.person_effects[effect_name]["active"] = self.person_effect_vars[effect_name].get()
        else:
            self.video_processor.background_effects[effect_name]["active"] = self.background_effect_vars[effect_name].get()
        # Efekt durumu değiştiğinde parametre slider'larını güncelle
        self.create_parameter_sliders()
    
    def update_video(self):
        frame = self.video_processor.process_frame()
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Görüntüyü pencere boyutuna göre yeniden boyutlandır
            display_size = (800, 600)  # Daha büyük görüntü boyutu
            img = img.resize(display_size, Image.Resampling.LANCZOS)
            
            # CTkImage kullanarak görüntüyü oluştur
            ctk_image = CTkImage(light_image=img, dark_image=img, size=display_size)
            self.video_label.configure(image=ctk_image)
            self.status_label.configure(text="Kamera durumu: Bağlı")
        else:
            self.video_label.configure(text="Kamera görüntüsü alınamıyor!", image=None)
            self.status_label.configure(text="Kamera durumu: Bağlantı hatası!")
        
        self.root.after(10, self.update_video)
        
    def update_view_mode(self):
        self.video_processor.view_mode = self.mode_var.get()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    processor = VideoProcessor()
    gui = ModernGUI(processor)
    gui.run()
    processor.release()
