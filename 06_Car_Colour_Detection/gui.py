import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from detect import detect_objects

selected_image = None

def display_image(path, canvas):
    img = Image.open(path)

    img.thumbnail((550, 400))

    photo = ImageTk.PhotoImage(img)

    canvas.delete("all")
    canvas.create_image(
        275,
        200,
        image=photo,
        anchor="center"
    )

    canvas.image = photo

def upload_image():
    global selected_image

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if not file_path:
        return

    selected_image = file_path

    display_image(file_path, input_canvas)

    output_canvas.delete("all")

    result_label.config(text="")


def detect_image():

    global selected_image

    if selected_image is None:
        messagebox.showwarning(
            "Warning",
            "Please upload an image first."
        )
        return

    output_path, cars, people = detect_objects(selected_image)

    display_image(output_path, output_canvas)

    result_label.config(
        text=f"Cars Detected : {cars}        People Detected : {people}"
    )

root = tk.Tk()

root.title("Car Colour Detection and Traffic Monitoring")

root.geometry("1300x750")

root.configure(bg="white")


title = tk.Label(
    root,
    text="Car Colour Detection and Traffic Monitoring",
    font=("Arial", 28, "bold"),
    fg="navy",
    bg="white"
)

title.pack(pady=20)


button_frame = tk.Frame(root, bg="white")

button_frame.pack()


upload_btn = tk.Button(
    button_frame,
    text="Upload Image",
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    width=15,
    command=upload_image
)

upload_btn.grid(row=0, column=0, padx=15)


detect_btn = tk.Button(
    button_frame,
    text="Detect",
    font=("Arial", 14, "bold"),
    bg="blue",
    fg="white",
    width=15,
    command=detect_image
)

detect_btn.grid(row=0, column=1, padx=15)

image_frame = tk.Frame(root, bg="white")

image_frame.pack(pady=20)


left_frame = tk.Frame(image_frame, bg="white")

left_frame.grid(row=0, column=0, padx=20)


right_frame = tk.Frame(image_frame, bg="white")

right_frame.grid(row=0, column=1, padx=20)


tk.Label(
    left_frame,
    text="Input Image",
    font=("Arial", 18, "bold"),
    bg="white"
).pack(pady=10)


input_canvas = tk.Canvas(
    left_frame,
    width=550,
    height=400,
    bg="#DDDDDD",
    highlightbackground="black",
    highlightthickness=2
)

input_canvas.pack()


tk.Label(
    right_frame,
    text="Output Image",
    font=("Arial", 18, "bold"),
    bg="white"
).pack(pady=10)


output_canvas = tk.Canvas(
    right_frame,
    width=550,
    height=400,
    bg="#DDDDDD",
    highlightbackground="black",
    highlightthickness=2
)

output_canvas.pack()

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 20, "bold"),
    fg="red",
    bg="white"
)

result_label.pack(pady=20)


root.mainloop()