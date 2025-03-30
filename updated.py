import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog,Canvas,PhotoImage
from PIL import Image, UnidentifiedImageError
from PIL import ImageTk
import time
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import os

# Initialize YOLO model
model = YOLO(r".\detection\weights\best.pt")

# App initialization
app = ctk.CTk()
app.geometry("950x650")
app.title("Detection and Classification Application")

# Variables
file_path_var = ctk.StringVar()
class_var = ctk.StringVar()
bolt_hook_var = ctk.StringVar(value="Bolt")

# UI Frame Containers
frame_top = ctk.CTkFrame(app, fg_color="#3b3b3b")
frame_top.pack(pady=10, padx=10, fill="x")

frame_body = ctk.CTkFrame(app)
frame_body.pack(fill="both", expand=True, pady=10, padx=10)

# Placeholder references for widgets
progress_bar = None
output_text = None
image_label = None
result_box = None

# Define variables
detected_offset = 32
line_position = 600  # Adjust line position as needed
frame_rate = 30
process_every_n_frames = 2 # Process every 2nd frame.

# UI Control Functions
def clear_frame(frame):
    global output_text, image_label
    for widget in frame.winfo_children():
        widget.destroy()
    output_text = None
    image_label = None

def toggle_mode(mode):
    clear_frame(frame_body)
    if mode == "object":
        object_btn.configure(state="disabled", fg_color="green")
        classify_btn.configure(state="normal", fg_color="blue")
        load_object_detection_ui()
    elif mode == "classify":
        classify_btn.configure(state="disabled", fg_color="green")
        object_btn.configure(state="normal", fg_color="blue")
        load_classification_ui()

def upload_file():
    filepaths = filedialog.askopenfilenames(filetypes=[("Media Files", "*.mp4 *.png *.jpg *.jpeg")])
    if filepaths:
        file_path_var.set(";".join(filepaths))  # Join multiple paths with a delimiter
        # Clear the result box on new upload
        if result_box and result_box.winfo_exists():
            result_box.delete("1.0", "end")
        if output_text and output_text.winfo_exists():
            output_text.delete("1.0", "end")
        if progress_bar:
            progress_bar.set(0)
            app.update()
            for i in range(101):
                time.sleep(0.005)
                progress_bar.set(i / 100)
                app.update()
        # Show preview of all selected files (images/videos)
        for filepath in filepaths:
            display_preview(filepath)
            break  # Just preview the first one for now

def display_preview(filepath):
    global image_label
    # Clear previous image preview
        # **Destroy previous image_label if it exists**
    if image_label is not None and image_label.winfo_exists():
        image_label.destroy()
        image_label = None  
    if image_label is None:
        image_label = ctk.CTkLabel(frame_body, text="Image Preview Here", width=400, height=250)
        image_label.pack(pady=10)
    if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(filepath)
        image.thumbnail((600, 400))
        photo = ctk.CTkImage(dark_image=image, size=(600, 400))
        image_label.configure(image=photo, text="")
        image_label.image = photo
    elif filepath.lower().endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            max_width, max_height = 600, 400  # UI dimensions

            display_image(filepath)  # Pass your Tkinter canvas widget

            photo = ctk.CTkImage(dark_image=image, size=(600, 400))
            image_label.configure(image=photo, text="")
            image_label.image = photo
        cap.release()

def run_detection():
    filepath = file_path_var.get()
    selected_class = class_var.get()
    detect_all = detect_all_var.get()  # Get checkbox state

    if not filepath:
        return

    if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(filepath, selected_class, detect_all)
    elif filepath.lower().endswith(('.mp4', '.avi')):
        process_video(filepath, selected_class, detect_all)

def process_image(filepath, selected_class, detect_all):
    global output_text, image_label, model
    if output_text is None or image_label is None:
        return
    if not output_text.winfo_exists() or not image_label.winfo_exists():
        return
    image = Image.open(filepath)
    results = model(filepath)

    # **Check if there are any detections**
    if results[0].boxes is None or len(results[0].boxes) == 0:  
        update_output_text_box("No objects detected")
        return  # Stop execution if no results

    detected_objects = {}
    
    for r in results:
        for c in r.boxes.cls:
            class_name = model.names[int(c)]
            if detect_all or class_name == selected_class:
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

    output_text.delete("1.0", "end")

    if detect_all:
        for class_name, count in detected_objects.items():
            output_text.insert("end", f"{class_name}: {count}\n")
    else:
        count = detected_objects.get(selected_class, 0)  # Show only selected class count
        output_text.insert("end", f"{selected_class}: {count}\n")

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))

        display_image(filepath)  # Pass your Tkinter canvas widget

        photo = ctk.CTkImage(light_image=im, size=(600, 400))

        if image_label is None or not image_label.winfo_exists():
            return
        image_label.configure(image=photo, text="")
        image_label.image = photo

def display_image(filepath):
    # Check if file is an image or video
    valid_image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    valid_video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in valid_image_extensions:
        try:
            im = Image.open(filepath)
        except UnidentifiedImageError:
            print(f"Error: Could not identify image file {filepath}")

    elif ext in valid_video_extensions:
        print(f"Detected video file: {filepath}. Extracting first frame...")

        # Open video with OpenCV
        cap = cv2.VideoCapture(filepath)
        success, frame = cap.read()
        cap.release()

        if success:
            # Convert BGR to RGB for correct colors
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
        else:
            print(f"Error: Could not read video file {filepath}")

    else:
        print(f"Error: Unsupported file type {ext}")

def process_video(filepath, selected_class, detect_all):
    def video_processing_thread():
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return

        detected_objects = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every_n_frames == 0:
                results = model(frame, conf=0.5) 
                
                # Get frame dimensions
                frame_height, frame_width, _ = frame.shape

                # Resize frame while maintaining aspect ratio
                max_width, max_height = 300, 400  # UI frame size
                scale = min(max_width / frame_width, max_height / frame_height)
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                cv2.line(frame, (0, line_position), (frame_width, line_position), (255, 127, 0), 3)

                detected = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        class_id = box.cls[0]
                        class_name = model.names[int(class_id)]

                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if detect_all or class_name.lower().strip() == selected_class.lower().strip(): 
                            detected.append((cx, cy, class_name))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check if detected objects are within the counting zone
                for (x, y, class_name) in detected:
                    if (line_position - detected_offset <= y <= line_position + detected_offset) and (300 <= x <= 700):
                        detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

                # Convert frame to PIL Image for Tkinter
                im_array = results[0].plot() if results else frame_resized
                im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
                photo = ctk.CTkImage(light_image=im, size=(300, 400))

                # Ensure image_label exists before updating
                if image_label and image_label.winfo_exists():
                    app.after(0, lambda: image_label.configure(image=photo, text=""))  

                # Update output text in GUI thread
                app.after(0, lambda: update_output_text(detected_objects, detect_all, selected_class))

                time.sleep(1 / (frame_rate / process_every_n_frames))

            frame_count += 1

        cap.release()

    threading.Thread(target=video_processing_thread).start()

def update_output_text(detected_objects, detect_all=True, selected_class=None):
    output_text.delete("1.0", "end")
    if detect_all:
        for class_name, count in detected_objects.items():
            output_text.insert("end", f"{class_name}: {count}\n")
    else:
        count = detected_objects.get(selected_class, 0)
        output_text.insert("end", f"{selected_class}: {count}\n")

def update_output_text_box(text):
    if output_text is None or not output_text.winfo_exists():
        return  # Prevent crashes if result_box is not initialized

    output_text.delete("1.0", "end")
    output_text.insert("end", text)

def detect_bolt(image_path):
    model_bolt = YOLO(r".\bolt_weights\best.pt")

    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to read the image. Please check the path.")
        return

    # Get prediction results
    results = model_bolt(image)

    # Extract the predicted class with the highest probability
    if len(results) > 0:
        names_dict = results[0].names  # Retrieve the mapping of class IDs to names
        probs = results[0].probs.data.tolist()  # Get the probabilities for all classes
        class_name = names_dict[np.argmax(probs)]  # Get the class with the highest probability
    else:
        class_name = "No Prediction"

    # Print the prediction result
    print(f"Defective status: {class_name}")

    return class_name

def detect_hook(image_path):
    model_hook = YOLO(r".\hook_weights\best.pt")

    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to read the image. Please check the path.")
        return

    # Get prediction results
    results = model_hook(image)

    # Extract the predicted class with the highest probability
    if len(results) > 0:
        names_dict = results[0].names  # Retrieve the mapping of class IDs to names
        probs = results[0].probs.data.tolist()  # Get the probabilities for all classes
        class_name = names_dict[np.argmax(probs)]  # Get the class with the highest probability
    else:
        class_name = "No Prediction"

    # Print the prediction result
    print(f"Defective status: {class_name}")

    return class_name

def toggle_radio(selection, mode):
    result = None
    global image_label

    # **Check if image_label exists before destroying it**
    if image_label is not None and image_label.winfo_exists():
        image_label.destroy()
        image_label = None  # Reset to None
    # Clear result box before updating
    update_result_box("Processing\n")

    if mode == "file":
        image_path = filedialog.askopenfilename()
        if not image_path:
            return
        display_preview(image_path)
        if selection == "Bolt":
            result = detect_bolt(image_path)
        elif selection == "Hook":
            result = detect_hook(image_path)

        app.after(0, lambda: update_result_box(f"Defective status: {result}\n"))

    elif mode == "folder":
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        defective_count = 0
        non_defective_count = 0

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(folder_path, filename)
                
                if selection == "Bolt":
                    result = detect_bolt(file_path)
                elif selection == "Hook":
                    result = detect_hook(file_path)

                if result == "Defective":
                    defective_count += 1
                else:
                    non_defective_count += 1

        summary = f"Total Images: {defective_count + non_defective_count}\nDefective: {defective_count}\nNon-Defective: {non_defective_count}\n"
        app.after(0, lambda: update_result_box(summary))

def update_result_box(text):
    if result_box and result_box.winfo_exists():
        result_box.delete("1.0", "end")
        result_box.insert("end", text)
    # Clear previous image
    if image_label and image_label.winfo_exists():
        image_label.configure(image=None, text="Image Results Here")

def load_object_detection_ui():
    global progress_bar, output_text, image_label, detect_all_var

    upload_btn = ctk.CTkButton(frame_body, text="Upload File", command=upload_file)
    upload_btn.pack(pady=10)

    progress_bar = ctk.CTkProgressBar(frame_body)
    progress_bar.pack(pady=5)
    progress_bar.set(0)

    # Detect All Checkbox
    detect_all_var = ctk.BooleanVar(value=False)

    def toggle_dropdown():
        """Enables/disables the dropdown based on the checkbox state."""
        if detect_all_var.get():
            dropdown.configure(state="disabled")
        else:
            dropdown.configure(state="normal")

    detect_all_checkbox = ctk.CTkCheckBox(
        frame_body, text="Detect All", variable=detect_all_var, command=toggle_dropdown
    )
    detect_all_checkbox.pack(pady=5)

    # Label for Dropdown
    dropdown_label = ctk.CTkLabel(frame_body, text="Select Class:", font=("Arial", 14, "bold"))
    dropdown_label.pack(pady=(10, 0))  # Add spacing above the label

    # Dropdown for Class Selection
    dropdown = ctk.CTkComboBox(
        frame_body,
        values=["Animals", "DefectiveFasteners", "Vegetation", "FishplateIssue", "BrokenRail",
                "DefectiveFastenerIndia", "SleeperCrack"],
        variable=class_var,
        state="normal"
    )
    dropdown.pack(pady=10)

    run_btn = ctk.CTkButton(frame_body, text="Run Detection", command=run_detection)
    run_btn.pack(pady=10)

    output_text = ctk.CTkTextbox(frame_body, width=400, height=200)
    output_text.pack(pady=10)

    image_label = ctk.CTkLabel(frame_body, text="Image/Video Output Here", width=400, height=250)
    image_label.pack(pady=10)

def load_classification_ui():
    global progress_bar, image_label, result_box

    radio_bolt = ctk.CTkRadioButton(
        frame_body, text="Bolt", variable=bolt_hook_var, value="Bolt", command=lambda: toggle_radio("Bolt", "file")
    )
    radio_hook = ctk.CTkRadioButton(
        frame_body, text="Hook", variable=bolt_hook_var, value="Hook", command=lambda: toggle_radio("Hook", "file")
    )

    radio_bolt.pack(pady=5)
    radio_hook.pack(pady=5)

    # Process a single image
    process_file_btn = ctk.CTkButton(
        frame_body, text="Process Image", command=lambda: toggle_radio(bolt_hook_var.get(), "file")
    )
    process_file_btn.pack(pady=10)

    # Process all images in a folder
    process_folder_btn = ctk.CTkButton(
        frame_body, text="Process Folder", command=lambda: toggle_radio(bolt_hook_var.get(), "folder")
    )
    process_folder_btn.pack(pady=10)

    result_box = ctk.CTkTextbox(frame_body, width=300, height=100)
    result_box.pack(pady=10)

    image_label = ctk.CTkLabel(frame_body, text="Image Results Here", width=400, height=250)
    image_label.pack(pady=10)


# Buttons
object_btn = ctk.CTkButton(frame_top, text="Object Detection", fg_color="green", command=lambda: toggle_mode("object"))
object_btn.pack(side="left", padx=10)

classify_btn = ctk.CTkButton(frame_top, text="Classification", fg_color="blue", command=lambda: toggle_mode("classify"))
classify_btn.pack(side="left", padx=10)

# Default UI Mode
toggle_mode("object")

app.mainloop()