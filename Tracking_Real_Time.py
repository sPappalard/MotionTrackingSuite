import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import time

class TrackingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Enhanced Tracking Application")
        self.window.geometry("1000x800")
        
        # Inizializzazione modelli MediaPipe
        self.init_models()
        
        # Variabili per il tracking
        self.is_tracking = False
        self.tracking_type = None
        self.cap = None
        self.current_object_model = 'Chair'
        
        self.create_menu()
        
    def init_models(self):
        # Modelli che vengono inizializzati all'avvio
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Inizializzazione oggetti di drawing
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def create_menu(self):
        for widget in self.window.winfo_children():
            widget.destroy()
            
        style = ttk.Style()
        style.configure('Menu.TButton', font=('Helvetica', 12), padding=10)
        
        menu_frame = ttk.Frame(self.window, padding="20")
        menu_frame.pack(expand=True)
        
        ttk.Label(menu_frame, text="Seleziona Modalità di Tracking", 
                 font=('Helvetica', 14, 'bold')).pack(pady=20)
        
        tracking_options = [
            ("Tracking Mani", "hands"),
            ("Tracking Viso (Mesh)", "face"),
            ("Rilevamento Volto (Bounding Box)", "face_detection"),
            ("Segmentazione Corpo/Capelli", "segmentation"),
            ("Tracking Pose", "pose"),
            ("Tracking Oggetti 3D", "object"),
            ("Tracking Olistico", "holistic")
        ]
        
        for text, tracking_type in tracking_options:
            ttk.Button(menu_frame, text=text, 
                      command=lambda t=tracking_type: self.start_tracking(t),
                      style='Menu.TButton').pack(pady=10, padx=20, fill='x')

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.tracking_type == "hands":
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.drawing_styles.get_default_hand_landmarks_style(),
                        self.drawing_styles.get_default_hand_connections_style()
                    )
                    
        elif self.tracking_type == "face":
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_styles
                        .get_default_face_mesh_contours_style()
                    )

        elif self.tracking_type == "face_detection":
            results = self.face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    self.mp_draw.draw_detection(frame, detection)

        elif self.tracking_type == "segmentation":
            results = self.selfie_segmentation.process(rgb_frame)
            if results.segmentation_mask is not None:
                mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.2
                bg_image = np.zeros_like(frame)
                frame = np.where(mask, frame, bg_image)

        elif self.tracking_type == "pose":
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
                )

        elif self.tracking_type == "object":
            if hasattr(self, 'objectron') and self.objectron:
                results = self.objectron.process(rgb_frame)
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        self.mp_draw.draw_landmarks(
                            frame, 
                            detected_object.landmarks_2d, 
                            self.mp_objectron.BOX_CONNECTIONS,
                            self.drawing_styles.get_default_pose_landmarks_style()
                        )

        elif self.tracking_type == "holistic":
            results = self.holistic.process(rgb_frame)
            
            # Disegna tutti i componenti
            self.mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles
                .get_default_pose_landmarks_style()
            )
            if results.left_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )
            if results.right_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )
        
        return frame

    def start_tracking(self, tracking_type):
        self.tracking_type = tracking_type
        self.is_tracking = True
        
        # Inizializza Objectron se necessario
        if tracking_type == "object":
            self.mp_objectron = mp.solutions.objectron
            self.objectron = self.mp_objectron.Objectron(
                static_image_mode=False,
                max_num_objects=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_name=self.current_object_model
            )
        
        for widget in self.window.winfo_children():
            widget.destroy()
            
        main_frame = ttk.Frame(self.window)
        main_frame.pack(expand=True, fill='both')
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=5, padx=5)
        
        style = ttk.Style()
        style.configure('Back.TButton', font=('Helvetica', 10, 'bold'))
        
        back_button = ttk.Button(
            button_frame, 
            text="← Torna al Menu", 
            command=self.stop_tracking,
            style='Back.TButton'
        )
        back_button.pack(side='left', padx=5)
        
        # Aggiungi selezione modello per Objectron
        if tracking_type == "object":
            self.model_var = tk.StringVar(value='Chair')
            model_menu = ttk.Combobox(
                button_frame, 
                textvariable=self.model_var,
                values=['Chair', 'Shoe', 'Cup', 'Camera'],
                state='readonly'
            )
            model_menu.pack(side='left', padx=5)
            model_menu.bind('<<ComboboxSelected>>', self.change_object_model)
        
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.pack(expand=True, fill='both', pady=5)
        
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
        
    def change_object_model(self, event):
        self.current_object_model = self.model_var.get()
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name=self.current_object_model
        )
        
    def stop_tracking(self):
        self.is_tracking = False
        if self.cap is not None:
            self.cap.release()
        if hasattr(self, 'objectron'):
            self.objectron = None
        self.create_menu()
    
    def update_frame(self):
        if self.is_tracking and self.cap is not None:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)
                
                # Calcola e mostra gli FPS
                fps = 1 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                
                img.thumbnail((1000, 800), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                
                self.window.after(10, self.update_frame)

def main():
    root = tk.Tk()
    app = TrackingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()