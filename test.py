import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import pyautogui
import time

class UltimateTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate AI Tracking Suite")
        self.root.geometry("1280x960")
        
        # Inizializzazione modelli MediaPipe
        self.init_mediapipe_models()
        
        # Variabili di stato
        self.is_tracking = False
        self.cap = None
        self.tracking_mode = None
        self.mouse_control = False
        self.show_fps = True
        self.current_effect = None
        
        # Configurazione stili
        self.configure_styles()
        self.create_main_menu()

    def configure_styles(self):
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TButton', 
                           font=('Helvetica', 10, 'bold'),
                           padding=6,
                           background='#3498db',
                           foreground='white')
        self.style.map('TButton', 
                     background=[('active', '#2980b9')],
                     foreground=[('active', 'white')])

    def init_mediapipe_models(self):
        """Inizializza tutti i modelli MediaPipe"""
        # Modelli principali
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.mp_objectron = mp.solutions.objectron
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name='Cup')
        
        # Utilità di disegno
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def create_main_menu(self):
        """Crea il menu principale con tutte le opzioni"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill='both')
        
        title = ttk.Label(main_frame, 
                        text="AI Tracking Suite", 
                        font=('Helvetica', 24, 'bold'),
                        foreground='#ecf0f1',
                        background='#2c3e50')
        title.pack(pady=20)
        
        tracking_modes = [
            ("👐 Tracking Mani", 'hands'),
            ("😀 Rilevamento Volto", 'face'),
            ("💃 Tracking Postura", 'pose'),
            ("☕ Rilevamento Oggetti 3D", 'object'),
            ("🖱️ Controllo Mouse", 'mouse'),
            ("🎭 Effetti AR", 'ar'),
            ("📐 Analisi Postura", 'posture'),
            ("📊 Fitness Tracker", 'fitness'),
            ("🚫 Sistema Antiaffaticamento", 'fatigue'),
            ("🎮 Virtual Controller", 'controller')
        ]
        
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(expand=True)
        
        row, col = 0, 0
        for text, mode in tracking_modes:
            btn = ttk.Button(grid_frame, text=text,
                           command=lambda m=mode: self.start_tracking(m),
                           style='TButton')
            btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            col += 1
            if col > 2:
                col = 0
                row += 1

    def start_tracking(self, mode):
        """Avvia la modalità di tracking selezionata"""
        self.tracking_mode = mode
        self.is_tracking = True
        self.clear_window()
        
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)
        
        # Pannello controllo
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(fill='x', side='top')
        
        ttk.Button(control_frame, text="🔙 Menu Principale",
                 command=self.stop_tracking).pack(side='left')
        
        if self.tracking_mode == 'object':
            self.object_model_var = tk.StringVar(value='Cup')
            object_menu = ttk.Combobox(control_frame, 
                                     textvariable=self.object_model_var,
                                     values=['Cup', 'Shoe', 'Chair', 'Camera'],
                                     state='readonly')
            object_menu.pack(side='left', padx=10)
            object_menu.bind('<<ComboboxSelected>>', self.change_object_model)
        
        # Visualizzazione video
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(expand=True, fill='both')
        
        # Avvia acquisizione video
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def process_frame(self, frame):
        """Elabora il frame in base alla modalità selezionata"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.tracking_mode == 'hands':
            frame = self.process_hands(rgb_frame, frame)
        
        elif self.tracking_mode == 'face':
            frame = self.process_face(rgb_frame, frame)
        
        elif self.tracking_mode == 'pose':
            frame = self.process_pose(rgb_frame, frame)
        
        elif self.tracking_mode == 'object':
            frame = self.process_objects(rgb_frame, frame)
        
        elif self.tracking_mode == 'mouse':
            frame = self.control_mouse(rgb_frame, frame)
        
        elif self.tracking_mode == 'ar':
            frame = self.apply_ar_effects(rgb_frame, frame)
        
        elif self.tracking_mode == 'posture':
            frame = self.analyze_posture(rgb_frame, frame)
        
        if self.show_fps:
            frame = self.add_fps_counter(frame)
        
        return frame

    def process_hands(self, rgb_frame, output_frame):
        """Elaborazione per il tracking delle mani"""
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style())
        return output_frame

    def process_face(self, rgb_frame, output_frame):
        """Elaborazione per il tracking del volto"""
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles
                    .get_default_face_mesh_tesselation_style())
        return output_frame

    def process_pose(self, rgb_frame, output_frame):
        """Elaborazione per il tracking della postura"""
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles
                .get_default_pose_landmarks_style())
        return output_frame

    def process_objects(self, rgb_frame, output_frame):
        """Elaborazione per il rilevamento oggetti 3D"""
        results = self.objectron.process(rgb_frame)
        if results.detected_objects:
            for obj in results.detected_objects:
                self.mp_drawing.draw_landmarks(
                    output_frame, 
                    obj.landmarks_2d, 
                    self.mp_objectron.BOX_CONNECTIONS)
        return output_frame

    def control_mouse(self, rgb_frame, output_frame):
        """Controllo del mouse con le mani"""
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                screen_w, screen_h = pyautogui.size()
                x = int(index_tip.x * screen_w)
                y = int(index_tip.y * screen_h)
                pyautogui.moveTo(x, y)
                
                if self.is_fist_closed(landmarks):
                    pyautogui.click()
                    cv2.putText(output_frame, "CLICK", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return output_frame

    def is_fist_closed(self, landmarks):
        """Rileva se la mano è chiusa a pugno"""
        tip_ids = [4, 8, 12, 16, 20]  # Indici delle punte delle dita
        return all(landmarks.landmark[i].y > landmarks.landmark[i-1].y 
                 for i in tip_ids)

    def apply_ar_effects(self, rgb_frame, output_frame):
        """Applica effetti di realtà aumentata"""
        # Effetto specchio magico
        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blurred)
        output_frame = cv2.divide(gray, inverted_blur, scale=256.0)
        return cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

    def analyze_posture(self, rgb_frame, output_frame):
        """Analisi della postura con feedback visivo"""
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calcola angolo tra le spalle
            angle = self.calculate_shoulder_angle(left_shoulder, right_shoulder)
            color = (0,255,0) if 160 < angle < 200 else (0,0,255)
            
            cv2.putText(output_frame, f"Shoulder Angle: {angle:.1f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return output_frame

    def calculate_shoulder_angle(self, left, right):
        """Calcola l'angolo tra le spalle"""
        dx = right.x - left.x
        dy = right.y - left.y
        return np.degrees(np.arctan2(abs(dy), dx))

    def add_fps_counter(self, frame):
        """Aggiunge il contatore FPS al frame"""
        fps = 1 / (time.time() - self.last_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        self.last_time = time.time()
        return frame

    def update_frame(self):
        """Aggiorna il frame video"""
        if self.is_tracking and self.cap.isOpened():
            self.last_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                processed = self.process_frame(frame)
                img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                img = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = img
                self.video_label.configure(image=img)
            
            self.root.after(10, self.update_frame)

    def stop_tracking(self):
        """Interrompe il tracking e torna al menu"""
        self.is_tracking = False
        if self.cap:
            self.cap.release()
        self.create_main_menu()

    def clear_window(self):
        """Pulisce la finestra"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def change_object_model(self, event):
        """Cambia il modello per il rilevamento oggetti"""
        new_model = self.object_model_var.get()
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name=new_model)

if __name__ == "__main__":
    app = UltimateTrackingApp(tk.Tk())
    app.root.mainloop()