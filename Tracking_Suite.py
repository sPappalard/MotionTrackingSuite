import tkinter as tk
from tkinter import ttk
import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from PIL import Image, ImageTk
import pyautogui
import time


class AdvancedTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced AI Tracking Suite")
        self.root.geometry("1024x768")

        # Initialize MediaPipe models
        self.init_mediapipe_models()

        # State variables
        self.is_tracking = False
        self.cap = None
        self.tracking_mode = None
        self.show_fps = True

        # Configure UI styling
        self.configure_styles()
        self.create_main_menu()

    def configure_styles(self):
        """Configure modern UI styles"""
        self.style = ttk.Style()
        self.style.configure('MainFrame.TFrame', background='#1a1a1a')
        self.style.configure('Card.TFrame', background='#2d2d2d')
        self.style.configure('ModernButton.TButton',
                             font=('Helvetica', 11, 'bold'),
                             padding=12,
                             background='#007AFF')
        self.style.configure('Title.TLabel',
                             font=('Helvetica', 24, 'bold'),
                             foreground='#ffffff',
                             background='#1a1a1a')
        self.style.configure('Description.TLabel',
                             font=('Helvetica', 10),
                             foreground='#cccccc',
                             background='#2d2d2d',
                             wraplength=200)

    def init_mediapipe_models(self):
        """Initialize required MediaPipe models"""
        # Hand tracking for mouse control
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)

        # 3D object detection
        self.mp_objectron = mp.solutions.objectron
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name='Cup')

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def create_main_menu(self):
        """Create modern main menu interface"""
        self.clear_window()

        main_frame = ttk.Frame(self.root, style='MainFrame.TFrame')
        main_frame.pack(expand=True, fill='both', padx=30, pady=30)

        # Title
        title = ttk.Label(main_frame,
                          text="AI Tracking Suite",
                          style='Title.TLabel')
        title.pack(pady=(0, 30))

        # Tracking modes with descriptions
        tracking_modes = [
            ("🖱️ Gesture Mouse Control", 'mouse',
             "Control your computer mouse using hand gestures. Point with your index finger to move, close your fist to click."),
            ("🎯 3D Object Detection", 'object',
             "Detect and track various 3D objects in real-time. Supports common objects like cups, shoes, chairs, and cameras."),
            ("📏 Distance Estimation", 'distance',
             "Measure approximate distances between your hand and the detected objects. Useful for spatial interactions.")
        ]

        # Create cards for each mode
        cards_frame = ttk.Frame(main_frame, style='MainFrame.TFrame')
        cards_frame.pack(expand=True)

        for i, (text, mode, desc) in enumerate(tracking_modes):
            card = ttk.Frame(cards_frame, style='Card.TFrame', padding=15)
            card.grid(row=0, column=i, padx=15, pady=15)

            ttk.Button(card,
                       text=text,
                       command=lambda m=mode: self.start_tracking(m),
                       style='ModernButton.TButton').pack(pady=(0, 10))

            ttk.Label(card,
                      text=desc,
                      style='Description.TLabel').pack()

    def start_tracking(self, mode):
        """Initialize tracking mode"""
        self.tracking_mode = mode
        self.is_tracking = True
        self.clear_window()

        # Main tracking interface
        main_frame = ttk.Frame(self.root, style='MainFrame.TFrame')
        main_frame.pack(fill='both', expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame, style='Card.TFrame', padding=10)
        control_frame.pack(fill='x', side='top', padx=10, pady=10)

        ttk.Button(control_frame,
                   text="← Back to Menu",
                   command=self.stop_tracking,
                   style='ModernButton.TButton').pack(side='left')

        # Model selection for object and distance
        if mode in ['object', 'distance']:
            self.object_model_var = tk.StringVar(value='Cup')
            models = ['Cup', 'Shoe', 'Chair', 'Camera']
            object_menu = ttk.Combobox(control_frame,
                                       textvariable=self.object_model_var,
                                       values=models,
                                       state='readonly')
            object_menu.pack(side='left', padx=10)
            object_menu.bind('<<ComboboxSelected>>', self.change_object_model)
            # Update the initial template
            self.change_object_model(None)

        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(expand=True, fill='both', padx=10, pady=(0, 10))

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        """Reads a frame from the webcam, processes it and updates the display"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                # Converts the image for Tkinter
                cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        if self.is_tracking:
            self.root.after(10, self.update_frame)

    def process_frame(self, frame):
        """Process the frame according to the selected mode"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.tracking_mode == 'mouse':
            frame = self.process_mouse_control(rgb_frame, frame)
        elif self.tracking_mode == 'object':
            frame = self.process_object_detection(rgb_frame, frame)
        elif self.tracking_mode == 'distance':
            frame = self.process_distance_estimation(rgb_frame, frame)

        if self.show_fps:
            frame = self.add_fps_counter(frame)

        return frame

    def process_mouse_control(self, rgb_frame, output_frame):
        """Process hand tracking for mouse control"""
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style())

                # mouse control
                index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                screen_w, screen_h = pyautogui.size()
                x = int(index_tip.x * screen_w)
                y = int(index_tip.y * screen_h)
                pyautogui.moveTo(x, y)

                if self.is_fist_closed(landmarks):
                    pyautogui.click()
                    cv2.putText(output_frame, "Click!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return output_frame

    def process_object_detection(self, rgb_frame, output_frame):
        """Process 3D object detection"""
        results = self.objectron.process(rgb_frame)
        if results.detected_objects:
            for obj in results.detected_objects:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    obj.landmarks_2d,
                    self.mp_objectron.BOX_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style())

                # Adds the object label
                cv2.putText(output_frame,
                            f"Detected: {self.object_model_var.get()}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)
        return output_frame

    def process_distance_estimation(self, rgb_frame, output_frame):
        """Process the distance between hand and object"""
        # Process the hand
        hand_results = self.hands.process(rgb_frame)
        hand_position = None

        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style())

                # Gets the position of the index tip
                index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_position = (int(index_tip.x * output_frame.shape[1]),
                                 int(index_tip.y * output_frame.shape[0]))

        # Processes the objects
        object_results = self.objectron.process(rgb_frame)
        if object_results.detected_objects and hand_position:
            for obj in object_results.detected_objects:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    obj.landmarks_2d,
                    self.mp_objectron.BOX_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style())

                # Calculate the center of the object
                landmarks = np.array([[lm.x * output_frame.shape[1],
                                       lm.y * output_frame.shape[0]]
                                      for lm in obj.landmarks_2d.landmark])
                obj_center = np.mean(landmarks, axis=0).astype(int)

                # Draw a line between hand and object
                cv2.line(output_frame, hand_position, tuple(obj_center), (0, 255, 0), 2)

                # Calculate and display distance
                distance = np.sqrt(np.sum((np.array(hand_position) - obj_center) ** 2))
                distance_pixels = int(distance)

                cv2.putText(output_frame,
                            f"Distance: {distance_pixels}px",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)

                # Areas of interaction
                cv2.circle(output_frame, tuple(obj_center), 50, (0, 255, 0), 2)  # Zona vicina
                cv2.circle(output_frame, tuple(obj_center), 150, (0, 255, 255), 2)  # Zona media

                # Suggestion based on distance
                if distance_pixels < 50:
                    status = "Very Close"
                    color = (0, 255, 0)
                elif distance_pixels < 150:
                    status = "Medium Range"
                    color = (0, 255, 255)
                else:
                    status = "Far"
                    color = (0, 0, 255)

                cv2.putText(output_frame,
                            f"Status: {status}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2)

        return output_frame

    def add_fps_counter(self, frame):
        """Adds a FPS counter to the frame"""
        if not hasattr(self, 'prev_time'):
            self.prev_time = time.time()
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return frame

    def is_fist_closed(self, landmarks):
        """Simple heuristic to determine if the fist is closed"""
        # In this example, the fist is considered closed if the distance between the tip and the base of the index is small.
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        distance = np.sqrt((index_tip.x - index_mcp.x) ** 2 + (index_tip.y - index_mcp.y) ** 2)
        return distance < 0.05

    def stop_tracking(self):
        """Stop tracking and return to main menu"""
        self.is_tracking = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'prev_frame'):
            del self.prev_frame
        self.create_main_menu()

    def clear_window(self):
        """Cleans the window by removing all widgets"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def change_object_model(self, event):
        """Change the 3D survey model"""
        new_model = self.object_model_var.get()
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name=new_model)


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedTrackingApp(root)
    root.mainloop()
