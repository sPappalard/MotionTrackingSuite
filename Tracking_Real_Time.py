import tkinter as tk
from tkinter import ttk, messagebox
import cv2                              #for video capture and image processing
import mediapipe as mp                  #Google library for real time detection and tracking of hands, faces, poses and more
import numpy as np                      #provides maths functions
from PIL import Image, ImageTk          #Used for image manipulation and to convert frames into a tkinter-compatible format
import time
import threading
import os

#This  class handles a loading animation drawn on a tkinter Canvas object
class LoadingAnimation:
    #the constructor
    def __init__(self, canvas, width, height, color="#3498db"):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.color = color
        #to control if the animation is active
        self.is_running = False         
        #starting angle for the rotation of the animation
        self.angle = 0
        #ID of the animation (to stop the animation)
        self.animation_id = None

    #to start the animation    
    def start(self):
        self.is_running = True
        self.draw_frame()

    #to stop the animation    
    def stop(self):
        self.is_running = False
        #to clean the canvas
        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)
            self.canvas.delete("all")

    #to draw the animation's frame        
    def draw_frame(self):
        #if the animation is not running, exit to the function
        if not self.is_running:
            return
        
            
        self.canvas.delete("all")
        
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(center_x, center_y) - 10
        
        # Calculate positions for 8 arc points
        for i in range(8):
            # Calculate angle for this point
            angle = self.angle + i * 45
            alpha = min(255, 100 + i * 20)  # Opacity based on index
            color = self.calculate_color(self.color, alpha)
            
            # Convert to radians and calculate position
            rad = angle * np.pi / 180
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            
            # Draw the point
            size = 5 + i/2  # Increase size based on index
            self.canvas.create_oval(
                x-size, y-size, x+size, y+size, 
                fill=color, outline=""
            )
        
        # Update angle for next frame
        self.angle = (self.angle + 3) % 360
        
        # Schedule next frame
        self.animation_id = self.canvas.after(20, self.draw_frame)
    
    @staticmethod
    def calculate_color(hex_color, alpha):
        # Convert hex color to RGB with transparency
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f'#{r:02x}{g:02x}{b:02x}'

class ModernTheme:
    """Class for managing the application's modern theme"""
    PRIMARY = "#3498db"     # Primary blue
    SECONDARY = "#2ecc71"   # Accent green
    BACKGROUND = "#f8f9fa"  # Light background
    DARK_BG = "#343a40"     # Dark background
    TEXT = "#212529"        # Main text
    LIGHT_TEXT = "#f8f9fa"  # Light text
    
    @staticmethod
    def setup(root):
        # Configure application theme
        style = ttk.Style()
        
        # General configuration
        style.configure(".", 
                       font=('Segoe UI', 10),
                       background=ModernTheme.BACKGROUND)
        
        # Modern buttons
        style.configure('Modern.TButton', 
                       font=('Segoe UI', 11),
                       padding=10)
        
        style.map('Modern.TButton',
                 background=[('active', ModernTheme.PRIMARY), 
                            ('!active', "#fff")],
                 foreground=[('active', "#fff"), 
                            ('!active', ModernTheme.PRIMARY)])
        
        # Back button
        style.configure('Back.TButton', 
                       font=('Segoe UI', 10),
                       padding=8)
        
        # Frame
        style.configure('Card.TFrame', 
                       background="#fff",
                       relief="flat",
                       borderwidth=0)
        
        # Label for titles
        style.configure('Title.TLabel',
                       font=('Segoe UI', 16, 'bold'),
                       background="#fff",
                       foreground=ModernTheme.TEXT)
        
        # Label for subtitles
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 11),
                       background="#fff",
                       foreground="#6c757d")
        
        # Label for current mode
        style.configure('Mode.TLabel',
                       font=('Segoe UI', 10, 'bold'),
                       background=ModernTheme.BACKGROUND,
                       foreground=ModernTheme.PRIMARY)
        
        # Configure colors for camera canvas
        style.configure('Camera.TFrame',
                       background="#000")
        
        return style

class TrackingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Motion Tracking Suite")
        self.window.geometry("900x700")
        self.window.configure(bg=ModernTheme.BACKGROUND)
        self.window.minsize(800, 600)
        
        # Initialize modern theme
        self.style = ModernTheme.setup(self.window)
        
        # Variables for error control and recovery
        self.error_count = 0
        self.last_error_time = 0
        
        # Flag for close control
        self.is_closing = False
        
        # Initialize MediaPipe models
        self.init_models()
        
        # Tracking variables
        self.is_tracking = False
        self.tracking_type = None
        self.cap = None
        self.loading_animation = None
        
        # Resource management
        self.tracking_thread = None
        
        # Show main menu
        self.create_menu()
        
        # Configure close handling
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def init_models(self):
        try:
            # Initialize tracking models at startup
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
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                               f"Unable to initialize MediaPipe: {str(e)}")
            self.window.quit()

    def create_menu(self):
        # Clear existing widgets
        for widget in self.window.winfo_children():
            widget.destroy()
            
        # Create main container frame
        main_container = ttk.Frame(self.window, style='Card.TFrame')
        main_container.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Main title
        title_frame = ttk.Frame(main_container, style='Card.TFrame')
        title_frame.pack(fill='x', pady=(0, 20))
        
        title = ttk.Label(title_frame, 
                        text="Motion Tracking Suite",
                        style='Title.TLabel')
        title.pack(anchor='center', pady=10)
        
        subtitle = ttk.Label(title_frame,
                          text="Select a tracking mode",
                          style='Subtitle.TLabel')
        subtitle.pack(anchor='center')
        
        # Create frame for options grid
        options_frame = ttk.Frame(main_container, style='Card.TFrame')
        options_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Grid configuration
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        options_frame.rowconfigure(0, weight=1)
        options_frame.rowconfigure(1, weight=1)
        
        # Tracking options with icons and descriptions
        tracking_options = [
            ("Hand Tracking", "hands", "Track hand movements and gestures", 0, 0),
            ("Face Mesh", "face", "Detailed facial landmark tracking", 0, 1),
            ("Pose Estimation", "pose", "Full body posture tracking", 1, 0),
            ("Holistic Tracking", "holistic", "Combined face, pose and hand tracking", 1, 1)
        ]
        
        # Create buttons with descriptions
        for text, tracking_type, description, row, col in tracking_options:
            card = self.create_option_card(options_frame, text, description, tracking_type)
            card.grid(row=row, column=col, sticky='nsew', padx=10, pady=10)
    
    def create_option_card(self, parent, title, description, tracking_type):
        # Create a frame for the card
        card = ttk.Frame(parent, style='Card.TFrame')
        card.pack_propagate(False)
        
        # Add border and shadow with canvas
        canvas = tk.Canvas(card, highlightthickness=0, bg="white")
        canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Draw a thin border
        canvas.create_rectangle(
            2, 2, canvas.winfo_reqwidth()-2, canvas.winfo_reqheight()-2,
            outline="#e9ecef", width=1, fill="white"
        )
        
        # Card content
        content = ttk.Frame(card, style='Card.TFrame')
        content.pack(expand=True, fill='both', padx=15, pady=15)
        
        # Title
        ttk.Label(content, 
                 text=title,
                 font=('Segoe UI', 14, 'bold'),
                 background="white",
                 foreground=ModernTheme.TEXT).pack(anchor='w', pady=(0, 5))
        
        # Description
        ttk.Label(content,
                 text=description,
                 font=('Segoe UI', 10),
                 background="white",
                 foreground="#6c757d").pack(anchor='w', pady=(0, 15))
        
        # Canvas for loading animation
        loading_canvas = tk.Canvas(content, width=50, height=50, 
                                  highlightthickness=0, bg="white")
        loading_canvas.pack_forget()  # Initially hidden
        
        # Button
        button = ttk.Button(content, 
                          text="Start",
                          style='Modern.TButton',
                          command=lambda: self.handle_tracking_selection(
                              tracking_type, button, loading_canvas))
        button.pack(side='bottom')
        
        return card
    
    def handle_tracking_selection(self, tracking_type, button, loading_canvas):
        # Disable button
        button.configure(state='disabled')
        
        # Show and start loading animation
        loading_canvas.pack(pady=10)
        loading_animation = LoadingAnimation(loading_canvas, 50, 50, ModernTheme.PRIMARY)
        loading_animation.start()
        
        # Simulate loading
        def start_after_delay():
            time.sleep(1)  # Simulate loading
            self.window.after(0, lambda: self.start_tracking(tracking_type))
        
        # Start loading in a separate thread
        threading.Thread(target=start_after_delay, daemon=True).start()

    def process_frame(self, frame):
        try:
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
            
            # Add info overlay
            self.add_info_overlay(frame)
            
            return frame
        except Exception as e:
            # Log error but continue execution
            current_time = time.time()
            if current_time - self.last_error_time > 5:  # Limit error frequency
                self.error_count = 0
                self.last_error_time = current_time
            
            self.error_count += 1
            print(f"Error in frame processing: {str(e)}")
            
            # If too many consecutive errors, stop tracking
            if self.error_count > 10:
                self.window.after(0, self.stop_tracking)
                self.window.after(0, lambda: messagebox.showerror(
                    "Error", "Too many consecutive processing errors. The application will restart.")
                )
            
            # Return original frame in case of error
            return frame

    def add_info_overlay(self, frame):
        # Add overlay with FPS and current mode
        h, w = frame.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - getattr(self, 'last_frame_time', current_time)
        self.last_frame_time = current_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        
        # Semi-transparent area at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Current mode info
        mode_name = {
            "hands": "Hand Tracking",
            "face": "Face Mesh",
            "pose": "Pose Estimation",
            "holistic": "Holistic Tracking"
        }.get(self.tracking_type, self.tracking_type)
        
        # Add text
        cv2.putText(
            frame,
            f"{mode_name}",
            (10, h-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (w-100, h-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    def start_tracking(self, tracking_type):
        self.tracking_type = tracking_type
        self.is_tracking = True
        
        # Clear window and create new layout
        for widget in self.window.winfo_children():
            widget.destroy()
            
        # Create main layout
        main_frame = ttk.Frame(self.window, style='Card.TFrame')
        main_frame.pack(expand=True, fill='both', padx=15, pady=15)
        
        # Control bar at top
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Back button with modern style
        back_button = ttk.Button(
            control_frame, 
            text="← Back to Menu", 
            command=self.stop_tracking,
            style='Back.TButton'
        )
        back_button.pack(side='left')
        
        # Current mode indicator
        mode_name = {
            "hands": "Hand Tracking",
            "face": "Face Mesh",
            "pose": "Pose Estimation",
            "holistic": "Holistic Tracking"
        }.get(tracking_type, tracking_type)
        
        ttk.Label(
            control_frame,
            text=f"Mode: {mode_name}",
            style='Mode.TLabel'
        ).pack(side='right', padx=5)
        
        # Frame for video canvas with border
        video_container = ttk.Frame(main_frame, style='Camera.TFrame')
        video_container.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Canvas to display video
        self.video_canvas = tk.Canvas(
            video_container, 
            bg="black", 
            highlightthickness=0
        )
        self.video_canvas.pack(expand=True, fill='both')
        
        # Info block
        info_frame = ttk.Frame(main_frame, padding="10")
        info_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(
            info_frame,
            text="Position yourself in front of the webcam to start tracking",
            font=('Segoe UI', 9),
            foreground="#6c757d"
        ).pack(side='left')
        
        # Start capture in a separate thread
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Unable to open webcam")
            
            self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Error starting webcam: {str(e)}")
            self.stop_tracking()
    
    def stop_tracking(self):
        # Stop tracking and release resources
        self.is_tracking = False
        
        # Release webcam
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Return to main menu
        self.create_menu()
    
    def update_frame(self):
        if not self.is_tracking or self.cap is None or self.is_closing:
            return
            
        try:
            ret, frame = self.cap.read()
            
            if ret:
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Convert frame for display
                cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_image)
                
                # Resize to fit canvas while maintaining aspect ratio
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Make sure canvas is visible
                    # Calculate dimensions while maintaining aspect ratio
                    img_width, img_height = pil_img.size
                    ratio = min(canvas_width/img_width, canvas_height/img_height)
                    new_width = int(img_width * ratio)
                    new_height = int(img_height * ratio)
                    
                    pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                    
                # Convert to PhotoImage
                img_tk = ImageTk.PhotoImage(image=pil_img)
                
                # Update canvas
                self.video_canvas.config(width=canvas_width, height=canvas_height)
                self.video_canvas.create_image(
                    canvas_width//2, canvas_height//2,  # Center the image
                    image=img_tk, 
                    anchor=tk.CENTER
                )
                self.video_canvas.img_tk = img_tk  # Prevent garbage collection
                
                # Reset error counter if we have a valid frame
                self.error_count = 0
            else:
                # Handle case where there's no frame
                self.error_count += 1
                if self.error_count > 10:
                    messagebox.showerror("Error", "Unable to get frame from webcam")
                    self.stop_tracking()
                    return
            
            # Schedule next update
            self.window.after(15, self.update_frame)
            
        except Exception as e:
            # Error handling
            print(f"Error in frame update: {str(e)}")
            self.error_count += 1
            
            if self.error_count > 10:
                messagebox.showerror("Error", "Too many consecutive errors. Restarting application.")
                self.stop_tracking()
            else:
                # Retry
                self.window.after(100, self.update_frame)
    
    def on_closing(self):
        # Handle application closure
        self.is_closing = True
        self.is_tracking = False
        
        # Release resources
        if self.cap is not None:
            self.cap.release()
        
        # Close application
        self.window.destroy()

def main():
    try:
        # Verify dependencies are available
        required_modules = ["cv2", "mediapipe", "numpy", "PIL"]
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            # Show error message if modules are missing
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Dependency Error",
                f"The following modules are missing: {', '.join(missing_modules)}.\n"
                f"Install them with: pip install {' '.join(missing_modules)}"
            )
            root.destroy()
            return
        
        # Start application
        root = tk.Tk()
        app = TrackingApp(root)
        root.mainloop()
        
    except Exception as e:
        # Handle critical errors
        print(f"Critical error: {str(e)}")
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {str(e)}")
        root.destroy()

if __name__ == "__main__":
    main()