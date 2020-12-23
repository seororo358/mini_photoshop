import sys
import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
from PyQt5.QtCore import Qt

class mini_photoshop(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.label = QLabel()
        self.initUI()

    def initUI(self):
        self.label.setText('OpenCV Image')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('border: gray; border-style:solid; border-width: 1px;')

        btn_open = QPushButton('open')
        btn_open.clicked.connect(self.abrirImagen)

        btn_procesar = QPushButton('Processar Imagen')
        btn_procesar.clicked.connect(self.procesarImagen)

        btn_gray = QPushButton('gray')
        btn_gray.clicked.connect(self.img_to_gray)

        btn_histogram = QPushButton('histogram')
        btn_histogram.clicked.connect(self.histogram_processing)

        btn_sharpening1 = QPushButton('sharp1')
        btn_sharpening1.clicked.connect(self.sharpening1)
        btn_sharpening2 = QPushButton('sharp2')
        btn_sharpening2.clicked.connect(self.sharpening2)
        btn_sharpening3 = QPushButton('sharp3')
        btn_sharpening3.clicked.connect(self.sharpening3)

        btn_blur = QPushButton('blur1')
        btn_blur.clicked.connect(self.blurring_temp)
        #btn_undo = QPushButton('Undo')
        #btn_undo.clicked.connect(self.undo)

        top_bar = QHBoxLayout()
        top_bar.addWidget(btn_open)
        top_bar.addWidget(btn_procesar)
        top_bar.addWidget(btn_gray)
        top_bar.addWidget(btn_histogram)
        top_bar.addWidget(btn_sharpening1)
        top_bar.addWidget(btn_sharpening2)
        top_bar.addWidget(btn_sharpening3)
        top_bar.addWidget(btn_blur)
        #top_bar.addWidget(btn_undo)

        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addWidget(self.label)

        self.resize(1000, 750)
        self.setWindowTitle('OpenCV & PyQT 5 by Tutor de Programacion')

    def abrirImagen(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Buscar Imagen', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            with open(filename, "rb") as file:
                data = np.array(bytearray(file.read()))

                self.image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                # self.image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                self.mostrarImagen()

    def procesarImagen(self):
        if self.image is not None:
            # self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
            # self.image = cv2.Canny(self.image, 100, 200)

            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) \
                if len(self.image.shape) >= 3 else self.image

            blur = cv2.GaussianBlur(gray, (21, 21), 0, 0)

            self.image = cv2.divide(gray, blur, scale=256)
            self.mostrarImagen()

    def mostrarImagen(self):
        size = self.image.shape
        step = self.image.size / size[0]
        qformat = QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        self.resize(self.label.pixmap().size())

    def img_to_gray(self):
        if self.image is not None:
            #self.stacking(arr = self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray

            size = self.image.shape
            step = self.image.size / size[0]
            qformat = QImage.Format_Grayscale8

            img = QImage(self.image, size[1], size[0], step, qformat)

            self.label.setPixmap(QPixmap.fromImage(img))
            self.resize(self.label.pixmap().size())

    def histogram_processing(self):
        if self.image is not None:
            #self.stacking(arr = self.image)
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) \
                if len(self.image.shape) >= 3 else self.image

            clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(10, 10))
            img2 = clahe.apply(img)
            dst = np.hstack((img, img2))
            self.image = dst
            size = self.image.shape
            step = self.image.size / size[0]
            qformat = QImage.Format_Grayscale8
            his = QImage(self.image, size[1], size[0], step, qformat)

            self.label.setPixmap(QPixmap.fromImage(his))
            self.resize(self.label.pixmap().size())

    def sharpening1(self):
        if self.image is not None:
            #self.stacking(arr = self.image)
            img = self.image
            kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #sharpening filter
            output = cv2.filter2D(img, -1, kernel_sharpen_1)
            self.image = output
            self.mostrarImagen()

    def sharpening2(self):
        if self.image is not None:
            #self.stacking(arr = self.image)
            img = self.image
            kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])  #sharpening filter
            output = cv2.filter2D(img, -1, kernel_sharpen_2)
            self.image = output
            self.mostrarImagen()

    def sharpening3(self):
        if self.image is not None:
            img = self.image
            #self.stacking(arr=img)
            kernel_sharpen_3 = np.array(
                [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1],
                [-1, -1, -1, -1, -1]]) / 8.0   #sharpening filter
            output = cv2.filter2D(img, -1, kernel_sharpen_3)
            self.image = output
            self.mostrarImagen()

    def blurring_temp(self):
        #self.stacking(arr = self.image)
        clone = self.image.copy()
        rect_pts = []
        def blurring(event, x, y, flags, param):
            nonlocal rect_pts
            if self.image is not None:
                if event == cv2.EVENT_LBUTTONDOWN: #select the zone that drag mouse (blurring)
                    rect_pts = [(x,y)]
                elif event == cv2.EVENT_LBUTTONUP:
                    rect_pts.append((x,y))
                blurbox = clone[rect_pts[0][1]:rect_pts[1][1], rect_pts[0][0]:rect_pts[1][0]]
                clone[rect_pts[0][1]:rect_pts[1][1],rect_pts[0][0]:rect_pts[1][0]] = cv2.GaussianBlur(blurbox, (7,7), 1)
            else:
                pass
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',blurring)
        while True:
            cv2.imshow('image', clone)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                self.image = clone
                break
        self.mostrarImagen()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = mini_photoshop()
    win.show()
    sys.exit(app.exec_())
