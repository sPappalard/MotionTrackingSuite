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

        #cancel the previus canvas content  
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
        
        # Schedule next frame = use 'after' to call "draw_frame" again after 20 milliseconds thus creating the animation 
        self.animation_id = self.canvas.after(20, self.draw_frame)
    
    #to Convert hexadecimal color to RGB with transparency
    @staticmethod
    def calculate_color(hex_color, alpha):       
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f'#{r:02x}{g:02x}{b:02x}'

#This class defines the theme of the application
class ModernTheme:
    PRIMARY = "#3498db"     # Primary blue
    SECONDARY = "#2ecc71"   # Accent green
    BACKGROUND = "#f8f9fa"  # Light background
    DARK_BG = "#343a40"     # Dark background
    TEXT = "#212529"        # Main text
    LIGHT_TEXT = "#f8f9fa"  # Light text
    
    @staticmethod
    def setup(root):
        #Create a ttk Style object to configure the application theme
        style = ttk.Style()
        
        #Set a style for buttons
        style.configure(".", 
                       font=('Segoe UI', 10),
                       background=ModernTheme.BACKGROUND)
        
        #Modern buttons
        style.configure('Modern.TButton', 
                       font=('Segoe UI', 11),
                       padding=10)
        
        style.map('Modern.TButton',
                 background=[('active', ModernTheme.PRIMARY), 
                            ('!active', "#fff")],
                 foreground=[('active', "#fff"), 
                            ('!active', ModernTheme.PRIMARY)])
        
        #Back button
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

#This is the main class that runs the entire tracking application 
class TrackingApp:
    #to initialize the main window by setting title, initial size, background color and minimum size
    def __init__(self, window):
        self.window = window
        self.window.title("Motion Tracking Suite")
        self.window.geometry("900x700")
        self.window.configure(bg=ModernTheme.BACKGROUND)
        self.window.minsize(800, 600)
        
        #Initialize modern theme
        self.style = ModernTheme.setup(self.window)
        
        #Variables to control errors 
        self.error_count = 0
        self.last_error_time = 0
        
        #Flag for close control
        self.is_closing = False
        
        #Initializes the MediaPipe models for tracking (hands, face, posture and holistic tracking)
        self.init_models()
        
        #Variables to manage the tracking status, selected type, video capture, loading animation and thread for tracking
        self.is_tracking = False
        self.tracking_type = None
        self.cap = None
        self.loading_animation = None
        self.tracking_thread = None
        
        # Show main menu
        self.create_menu()
        
        #configure the window closing protocol, linking the event to the on_closing function
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    #to initialize models for tracking
    def init_models(self):
        try:
            #Initialize the hand tracking module 
            self.mp_hands = mp.solutions.hands
            #create an instance of the detector
            self.hands = self.mp_hands.Hands()
            
            #Initialize the face detector module 
            self.mp_face_mesh = mp.solutions.face_mesh
            #Set up the face landmark detector with parameters for maximum number of faces, detection confidence and tracking.            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            #Initialize the posture detector module 
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            #Initializes the holistic tracking module, which combines face, hands and posture
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            #Initialize utilities for drawing landmarks on images
            self.mp_draw = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                               f"Unable to initialize MediaPipe: {str(e)}")
            self.window.quit()

    #to create the main menu
    def create_menu(self):
        # Clear any existing widgets to create a new interface
        for widget in self.window.winfo_children():
            widget.destroy()
            
        #Create a main frame that will serve as the container for the menu
        main_container = ttk.Frame(self.window, style='Card.TFrame')
        main_container.pack(expand=True, fill='both', padx=20, pady=20)
        
        #A title and subtitle are created in the container, centered horizontally
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
        
        #A frame is created to arrange the tracking options in a grid with 2 columns and 2 rows
        options_frame = ttk.Frame(main_container, style='Card.TFrame')
        options_frame.pack(expand=True, fill='both', padx=10, pady=10)
        # Grid configuration
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        options_frame.rowconfigure(0, weight=1)
        options_frame.rowconfigure(1, weight=1)
        
        #A list of tuples that define each option: the display name, the tracking type, a description and the position (row, column) in the grid.
        tracking_options = [
            ("Hand Tracking", "hands", "Track hand movements and gestures", 0, 0),
            ("Face Mesh", "face", "Detailed facial landmark tracking", 0, 1),
            ("Pose Estimation", "pose", "Full body posture tracking", 1, 0),
            ("Holistic Tracking", "holistic", "Combined face, pose and hand tracking", 1, 1)
        ]
        
        #for each option, a 'card' is created using the "create_option_card" method and placed in the grid 
        for text, tracking_type, description, row, col in tracking_options:
            card = self.create_option_card(options_frame, text, description, tracking_type)
            card.grid(row=row, column=col, sticky='nsew', padx=10, pady=10)
    
    #to create a "card" frame that will contain title, description, loading animation and button.
    def create_option_card(self, parent, title, description, tracking_type):
        #Create a frame for the card
        card = ttk.Frame(parent, style='Card.TFrame')
        card.pack_propagate(False)
        
        #Add border and shadow with canvas
        canvas = tk.Canvas(card, highlightthickness=0, bg="white")
        canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        #Draw a thin border
        canvas.create_rectangle(
            2, 2, canvas.winfo_reqwidth()-2, canvas.winfo_reqheight()-2,
            outline="#e9ecef", width=1, fill="white"
        )
        
        #A content section with title and description is inserted inside the card
        #Card content
        content = ttk.Frame(card, style='Card.TFrame')
        content.pack(expand=True, fill='both', padx=15, pady=15)
        
        #Title
        ttk.Label(content, 
                 text=title,
                 font=('Segoe UI', 14, 'bold'),
                 background="white",
                 foreground=ModernTheme.TEXT).pack(anchor='w', pady=(0, 5))
        
        #Description
        ttk.Label(content,
                 text=description,
                 font=('Segoe UI', 10),
                 background="white",
                 foreground="#6c757d").pack(anchor='w', pady=(0, 15))
        
        #Create a canvas that will host the loading animation, initially hidden
        loading_canvas = tk.Canvas(content, width=50, height=50, 
                                  highlightthickness=0, bg="white")
        loading_canvas.pack_forget()  # Initially hidden
        
        #add "Start" button that, when clicked, calls "handle_tracking_selection" by passing tracking type, the button itself and the canvas of the animation
        button = ttk.Button(content, 
                          text="Start",
                          style='Modern.TButton',
                          command=lambda: self.handle_tracking_selection(
                              tracking_type, button, loading_canvas))
        button.pack(side='bottom')
        
        return card
    
    #to handle the tracking selection (simulate the loading animation after the click)
    def handle_tracking_selection(self, tracking_type, button, loading_canvas):
        #Disable button to avoid multiple clicks
        button.configure(state='disabled')
        
        #Show and start loading animation
        loading_canvas.pack(pady=10)
        loading_animation = LoadingAnimation(loading_canvas, 50, 50, ModernTheme.PRIMARY)
        loading_animation.start()
        
        # Simulate loading : Start a separate thread that simulates a delay (1 second) and then call start_tracking to start the selected tracking
        def start_after_delay():
            time.sleep(1) 
            self.window.after(0, lambda: self.start_tracking(tracking_type))
        # Start loading in a separate thread
        threading.Thread(target=start_after_delay, daemon=True).start()
    
    #to elaborate each frame captured from the webcam: based on the type of tracking selected (hands, face, posture or holistic), processes the image to detect and draw the appropriate landmarks
    def process_frame(self, frame):
        try:
            #Convert the frame from BGR (OpenCV standard) encoding to RGB, required for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #elaboration in according to the type of tracking
            #Use MediaPipe’s Hands model to process the frame and detect hand landmarks.If landmarks are found, draw them on the frame
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
            #Process the frame using the Face Mesh model to detect facial landmarks. If faces are detected, draw the face contours.
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
            #Use the MediaPipe's Pose model to detect body posture landmarks. If the posture landmarks are detected, it draws them.
            elif self.tracking_type == "pose":
                results = self.pose.process(rgb_frame)
                if results.pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
                    )

            #Use the Holistic model that combines face, posture and hand detection. For each component (face, posture, left and right hand), if the respective landmarks are present, draw them on the frame
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
            
            #After processing, add_info_overlay is called to add information (current mode and FPS) to the frame, and the processed frame is returned: to provide a visual feedback to the user
            self.add_info_overlay(frame)
            
            return frame
        #In case of Exceptions during the frame elaboration, an errors counter is increamented. If there are more then 10 errors, stop the tracking and show an error message    
        except Exception as e:
            #Log error but continue execution
            current_time = time.time()
            if current_time - self.last_error_time > 5:  # Limit error frequency
                self.error_count = 0
                self.last_error_time = current_time
            
            self.error_count += 1
            print(f"Error in frame processing: {str(e)}")
            
            #If too many consecutive errors, stop tracking
            if self.error_count > 10:
                self.window.after(0, self.stop_tracking)
                self.window.after(0, lambda: messagebox.showerror(
                    "Error", "Too many consecutive processing errors. The application will restart.")
                )
            
            #Return original frame in case of error
            return frame

    #to show a real time feedback (Tracking mode and FPS) to the user during the tracking 
    def add_info_overlay(self, frame):
        #Add overlay with FPS and current mode
        h, w = frame.shape[:2]
        
        # Calculate FPS: determine the time since the last frame to calculate FPS.
        current_time = time.time()
        elapsed_time = current_time - getattr(self, 'last_frame_time', current_time)
        self.last_frame_time = current_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        
        #Draw a Semi-transparent area at bottom to make the information readable 
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        #Current mode info
        mode_name = {
            "hands": "Hand Tracking",
            "face": "Face Mesh",
            "pose": "Pose Estimation",
            "holistic": "Holistic Tracking"
        }.get(self.tracking_type, self.tracking_type)
        
        #Add text: current mode 
        cv2.putText(
            frame,
            f"{mode_name}",
            (10, h-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        #Add text: FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (w-100, h-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    #to Set the selected tracking type and activate the tracking mode
    def start_tracking(self, tracking_type):
        self.tracking_type = tracking_type
        self.is_tracking = True
        
        # Clear window and create new layout dedicate to video viewing  
        for widget in self.window.winfo_children():
            widget.destroy()
            
        #Create main frame for the tracking screen 
        main_frame = ttk.Frame(self.window, style='Card.TFrame')
        main_frame.pack(expand=True, fill='both', padx=15, pady=15)
        
        #Create a control bar with a "back" button to return to the main menu, associating the stop_tracking function
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill='x', pady=(0, 10))

        back_button = ttk.Button(
            control_frame, 
            text="← Back to Menu", 
            command=self.stop_tracking,
            style='Back.TButton'
        )
        back_button.pack(side='left')
        
        #Display current mode next to "Back button"
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
        
        #Create a container for the video and a canvas where frames from the webcam will be displayed.
        video_container = ttk.Frame(main_frame, style='Camera.TFrame')
        video_container.pack(expand=True, fill='both', padx=5, pady=5)
        self.video_canvas = tk.Canvas(
            video_container, 
            bg="black", 
            highlightthickness=0
        )
        self.video_canvas.pack(expand=True, fill='both')
        
        # Info block: add a short instruction for the user
        info_frame = ttk.Frame(main_frame, padding="10")
        info_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(
            info_frame,
            text="Position yourself in front of the webcam to start tracking",
            font=('Segoe UI', 9),
            foreground="#6c757d"
        ).pack(side='left')
        
        # Try to open the webcam in a separate thread and, in case of success, start the "update_frame" function to start the capture loop
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Unable to open webcam")
            
            self.update_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Error starting webcam: {str(e)}")
            self.stop_tracking()
    
    #to stop tracking, relaase webcam resource and restore the main menu
    def stop_tracking(self):
        # Stop tracking and release resources
        self.is_tracking = False
        
        # Release webcam
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Return to main menu
        self.create_menu()
    
    #to capture each frame from the webacam, process it and show it on the GUI canvas 
    def update_frame(self):
        #if the tracking is not active, webcam is not available or the application is closing: the function exits
        if not self.is_tracking or self.cap is None or self.is_closing:
            return
            
        try:
            #read a frame from the webcam
            ret, frame = self.cap.read()
            
            #If the reading is successful:
            if ret:
                #Invert the frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                #Process frame based on the selected tracking mode
                frame = self.process_frame(frame)
                
                #Convert frame from BGR to RGB and turn it into a PIL image
                cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_image)
                
                #Resize the image to fit canvas while maintaining aspect ratio
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                # Make sure canvas is visible
                if canvas_width > 1 and canvas_height > 1:  
                    # Calculate dimensions while maintaining aspect ratio
                    img_width, img_height = pil_img.size
                    ratio = min(canvas_width/img_width, canvas_height/img_height)
                    new_width = int(img_width * ratio)
                    new_height = int(img_height * ratio)
                    pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                    
                #Convert the PIL image to a tkinter compatible format (PhotoImage) and draw it in the center of the canvas
                img_tk = ImageTk.PhotoImage(image=pil_img)
                self.video_canvas.config(width=canvas_width, height=canvas_height)
                self.video_canvas.create_image(
                    canvas_width//2, canvas_height//2,  
                    image=img_tk, 
                    anchor=tk.CENTER
                )
                #Prevent garbage collection: it is important to mantain a refer to the image
                self.video_canvas.img_tk = img_tk  
                
                #Reset error counter if we have a valid frame
                self.error_count = 0
            else:
                #handle case where there's no frame (increase the counter and, if there are more then 10 errors, stop tracking)
                self.error_count += 1
                if self.error_count > 10:
                    messagebox.showerror("Error", "Unable to get frame from webcam")
                    self.stop_tracking()
                    return
            
            #Schedule the recursive update_frame call after 15 milliseconds to continuously update the video
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
    
    #to handle the application closure
    def on_closing(self):
        self.is_closing = True
        self.is_tracking = False
        # Release resources
        if self.cap is not None:
            self.cap.release()
        # Close application
        self.window.destroy()

def main():
    try:
        #Verify dependencies are available
        required_modules = ["cv2", "mediapipe", "numpy", "PIL"]
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            #Show error message if modules are missing
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Dependency Error",
                f"The following modules are missing: {', '.join(missing_modules)}.\n"
                f"Install them with: pip install {' '.join(missing_modules)}"
            )
            root.destroy()
            return
        
        #if all dependencies are present

        #create the main window application 
        root = tk.Tk()
        #instantiate the TrackingApp
        app = TrackingApp(root)
        #starts the main loop of tkinter
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