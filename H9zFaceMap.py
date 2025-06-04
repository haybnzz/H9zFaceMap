import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
import random
import os

predictor_path = "h9z.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat (h9z.dat)")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def predict_emotion(avg_color):
    r, g, b = avg_color
    if r > g and r > b:
        return "Happy"
    elif b > r:
        return "Sad"
    return "Neutral"

def predict_race(avg_gray):
    if avg_gray < 90:
        return "African"
    elif avg_gray < 160:
        return "Asian"
    return "Caucasian"

class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition GUI")
        master.configure(bg="#1e1e1e")

        self.name_var = tk.StringVar()
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar()

        style = {"bg": "#1e1e1e", "fg": "#00ffff"}

        tk.Label(master, text="Name", **style).grid(row=0, column=0)
        tk.Entry(master, textvariable=self.name_var, bg="#2a2a2a", fg="#00ffff").grid(row=0, column=1)

        tk.Label(master, text="Age", **style).grid(row=1, column=0)
        tk.Entry(master, textvariable=self.age_var, bg="#2a2a2a", fg="#00ffff").grid(row=1, column=1)

        tk.Label(master, text="Gender", **style).grid(row=2, column=0)
        tk.Entry(master, textvariable=self.gender_var, bg="#2a2a2a", fg="#00ffff").grid(row=2, column=1)

        tk.Button(master, text="Upload Image", command=self.upload_image, bg="#2a2a2a", fg="#00ffff").grid(row=3, column=0, columnspan=2)
        tk.Button(master, text="Analyze Face", command=self.analyze_face, bg="#2a2a2a", fg="#00ffff").grid(row=4, column=0, columnspan=2)

        self.canvas = Canvas(master, width=800, height=600, scrollregion=(0, 0, 3000, 3000), bg="#1e1e1e")
        self.hbar = Scrollbar(master, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = Scrollbar(master, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.hbar.grid(row=6, column=0, columnspan=2, sticky='we')
        self.vbar.grid(row=5, column=2, sticky='ns')
        self.canvas.grid(row=5, column=0, columnspan=2)

        self.tk_img = None
        self.filepath = None

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.filepath = path
            img = Image.open(path)
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def analyze_face(self):
        if not self.filepath:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        name = self.name_var.get()
        age = self.age_var.get()
        gender = self.gender_var.get()

        img = cv2.imread(self.filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(img_rgb, 1)

        if len(dets) == 0:
            messagebox.showerror("Face Detection", "No face detected in the image.")
            return

        face = dets[0]
        shape = predictor(img_rgb, face)

        img_drawn = img_rgb.copy()

        # Rectangle and text
        cv2.rectangle(img_drawn, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(img_drawn, f"{name}, {age}, {gender}", (face.left(), max(face.top() - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Facial landmarks
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            dot_color = tuple(random.randint(50, 255) for _ in range(3))
            cv2.circle(img_drawn, (x, y), 1, dot_color, -1)

        def draw_line(start, end):
            for i in range(start, end):
                pt1 = (shape.part(i).x, shape.part(i).y)
                pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
                line_color = tuple(random.randint(100, 255) for _ in range(3))
                cv2.line(img_drawn, pt1, pt2, line_color, 1)

        draw_line(0, 16)
        draw_line(17, 21)
        draw_line(22, 26)
        draw_line(27, 30)
        draw_line(30, 35)
        draw_line(36, 41)
        draw_line(42, 47)
        draw_line(48, 59)
        draw_line(60, 67)
        cv2.line(img_drawn, (shape.part(41).x, shape.part(41).y), (shape.part(36).x, shape.part(36).y), (200, 200, 200), 1)
        cv2.line(img_drawn, (shape.part(47).x, shape.part(47).y), (shape.part(42).x, shape.part(42).y), (200, 200, 200), 1)

        # Emotion & Race
        face_crop = img_rgb[face.top():face.bottom(), face.left():face.right()]
        avg_color = np.mean(face_crop, axis=(0, 1))
        avg_gray = np.mean(cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY))
        emotion = predict_emotion(avg_color)
        race = predict_race(avg_gray)

        # Draw result labels and bars
        cv2.putText(img_drawn, f"Emotion: {emotion}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)
        cv2.putText(img_drawn, f"Race: {race}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

        cv2.rectangle(img_drawn, (150, 20), (150 + random.randint(40, 100), 25), (255, 200, 0), -1)
        cv2.rectangle(img_drawn, (150, 45), (150 + random.randint(40, 100), 50), (0, 200, 100), -1)

        final_img_bgr = cv2.cvtColor(img_drawn, cv2.COLOR_RGB2BGR)
        save_path = "output_face_analysis.jpg"
        cv2.imwrite(save_path, final_img_bgr)
        print(f"[✔] Saved analyzed image to: {save_path}")

        img_pil = Image.fromarray(img_drawn)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    FaceApp(root)
    root.mainloop()
