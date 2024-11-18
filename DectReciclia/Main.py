# Libraries
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math
import pygame

def clean_lbl():
    # Clean
    lblimg.config(image='')
    lblimgtxt.config(image='')

def images(img, imgtxt):
    img = img
    imgtxt = imgtxt

    # Img Detect
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

    # Img Text
    imgtxt = np.array(imgtxt, dtype="uint8")
    imgtxt = cv2.cvtColor(imgtxt, cv2.COLOR_BGR2RGB)
    imgtxt = Image.fromarray(imgtxt)

    img_txt = ImageTk.PhotoImage(image=imgtxt)
    lblimgtxt.configure(image=img_txt)
    lblimgtxt.image = img_txt


detected_classes = set()


def play_sound(cls):
    # Verificamos si la clase ya ha sido detectada para evitar reproducir el sonido nuevamente
    if cls not in detected_classes:
        detected_classes.add(cls)  # Marcamos la clase como detectada
        if cls == 0:
            pygame.mixer.Sound("C:/Users/pipe1/Downloads/Metalico.mp3").play()
        elif cls == 1:
            pygame.mixer.Sound("C:/Users/pipe1/Downloads/Vidrio.mp3").play()
        elif cls == 2:
            pygame.mixer.Sound("C:/Users/pipe1/Downloads/Plastico.mp3").play()
        elif cls == 3:
            
            pygame.mixer.Sound("C:/Users/pipe1/Downloads/Carton.mp3").play()
        elif cls == 4:
            pygame.mixer.Sound("C:/Users/pipe1/Downloads/Medico.mp3").play()


# Modificamos la función Scanning para restablecer las detecciones cuando no se detecta nada
def Scanning():
    global img_metal, img_glass, img_plastic, img_carton, img_medical
    global img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt, pantalla
    global lblimg, lblimgtxt, detected_classes

    # Interfaz
    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)

    lblimgtxt = Label(pantalla)
    lblimgtxt.place(x=995, y=310)
    detect = False

    # Read VideoCapture
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # True
        if ret == True:
            # Yolo | AntiSpoof
            results = model(frame, stream=True, verbose=False)
            for res in results:
                # Box
                boxes = res.boxes
                for box in boxes:
                    detect = True
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Error < 0
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    # Class
                    cls = int(box.cls[0])

                    # Confidence
                    conf = math.ceil(box.conf[0])

                    # Dibujar rectángulo y texto según la clase detectada
                    if cls == 0:
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        images(img_metal, img_metaltxt)
                        play_sound(cls)

                    elif cls == 1:
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        images(img_glass, img_glasstxt)
                        play_sound(cls)

                    elif cls == 2:
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        images(img_plastic, img_plastictxt)
                        play_sound(cls)

                    elif cls == 3:
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (150, 150, 150), 2)
                        images(img_carton, img_cartontxt)
                        play_sound(cls)

                    elif cls == 4:
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        images(img_medical, img_medicaltxt)
                        play_sound(cls)

            # Si no se detecta ninguna clase, restablecer la lista de clases detectadas
            if not detect:
                clean_lbl()
                detected_classes.clear()

            # Resize
            frame_show = imutils.resize(frame_show, width=640)

            # Convertimos el video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)

        else:
            cap.release()

# main
def ventana_principal():
    global cap, lblVideo, model, clsName, img_metal, img_glass, img_plastic, img_carton, img_medical
    global img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt, pantalla

    pygame.init()

    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

    # Background
    imagenF = PhotoImage(file="setUp/Canva.png")
    background = Label(image=imagenF, text="Inicio")
    background.place(x=0, y=0, relwidth=1, relheight=1)

    # Clases: 0 -> Metal | 1 -> Glass | 2 -> Plastic | 3 -> Carton | 4 -> Medical
    # Model
    model = YOLO('Modelos/best.pt')

    # Clases
    clsName = ['Metal', 'Glass', 'Plastic', 'Carton', 'Medical']

    # Images
    img_metal = cv2.imread("setUp/metal.png")
    img_glass = cv2.imread("setUp/vidrio.png")
    img_plastic = cv2.imread("setUp/plastico.png")
    img_carton = cv2.imread("setUp/carton.png")
    img_medical = cv2.imread("setUp/medical.png")
    img_metaltxt = cv2.imread("setUp/metaltxt.png")
    img_glasstxt = cv2.imread("setUp/vidriotxt.png")
    img_plastictxt = cv2.imread("setUp/plasticotxt.png")
    img_cartontxt = cv2.imread("setUp/cartontxt.png")
    img_medicaltxt = cv2.imread("setUp/medicaltxt.png")

    # Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=180)

    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Scanning()

    # Eject
    pantalla.mainloop()

if __name__ == "__main__":
    ventana_principal()
