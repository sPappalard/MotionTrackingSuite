import tkinter as tk
from tkinter import ttk
import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np
from PIL import Image, ImageTk
import time

class TrackingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Motion Tracking Suite")
        self.window.geometry("800x600")  # More compact window size
        
        # Initialize MediaPipe models
        self.init_models()
        
        # Tracking variables
        self.is_tracking = False
        self.tracking_type = None
        self.cap = None
        
        self.create_menu()
        
    def init_models(self):
        # Initialize tracking models on startup
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

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def create_menu(self):
        # Clear existing widgets
        for widget in self.window.winfo_children():
            widget.destroy()
            
        # Configure modern styling
        style = ttk.Style()
        style.configure('Menu.TButton', 
                       font=('Helvetica', 11),
                       padding=8)
        
        # Create centered menu frame
        menu_frame = ttk.Frame(self.window, padding="15")
        menu_frame.pack(expand=True)
        
        # Main title
        ttk.Label(menu_frame, 
                 text="Select Tracking Mode",
                 font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        # Tracking options with modern descriptions
        tracking_options = [
            ("Hand Tracking", "hands", "Track hand movements and gestures"),
            ("Face Mesh", "face", "Detailed facial landmark tracking"),
            ("Pose Estimation", "pose", "Full body pose tracking"),
            ("Holistic Tracking", "holistic", "Combined face, pose and hand tracking")
        ]
        
        # Create buttons with descriptions
        for text, tracking_type, description in tracking_options:
            option_frame = ttk.Frame(menu_frame)
            option_frame.pack(pady=8, padx=15, fill='x')
            
            ttk.Button(option_frame, 
                      text=text,
                      command=lambda t=tracking_type: self.start_tracking(t),
                      style='Menu.TButton').pack(fill='x')
            
            ttk.Label(option_frame,
                     text=description,
                     font=('Helvetica', 9)).pack(pady=(2, 0))

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

        elif self.tracking_type == "pose":
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
                )

        elif self.tracking_type == "holistic":
            results = self.holistic.process(rgb_frame)
            
            # Draw all holistic components
            if results.face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    self.mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
            if results.pose_landmarks:
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
        
        # Clear window and create new layout
        for widget in self.window.winfo_children():
            widget.destroy()
            
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(expand=True, fill='both')
        
        # Create control bar
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 5))
        
        # Style for back button
        style = ttk.Style()
        style.configure('Back.TButton', 
                       font=('Helvetica', 10, 'bold'))
        
        # Back button
        back_button = ttk.Button(
            control_frame, 
            text="← Back to Menu", 
            command=self.stop_tracking,
            style='Back.TButton'
        )
        back_button.pack(side='left')
        
        # Mode indicator
        ttk.Label(
            control_frame,
            text=f"Current Mode: {tracking_type.replace('_', ' ').title()}",
            font=('Helvetica', 10)
        ).pack(side='right', padx=5)
        
        # Video frame
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.pack(expand=True, fill='both')
        
        # Start capture
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
        
    def stop_tracking(self):
        self.is_tracking = False
        if self.cap is not None:
            self.cap.release()
        self.create_menu()
    
    def update_frame(self):
        if self.is_tracking and self.cap is not None:
            start_time = time.time()
            ret, frame = self.cap.read()
            
            if ret:
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)
                
                # Calculate and display FPS
                fps = 1 / (time.time() - start_time)
                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Convert frame for display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                
                # Scale image to fit window while maintaining aspect ratio
                img.thumbnail((780, 580), Image.Resampling.LANCZOS)
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