# --------------------------------------------------------------------------------------------------------------------------
#	LIBRARY IMPORTS
#   You need to install the following dependencies, use the following commands in a new terminal:
#  		* python -m pip install keras
#		* python -m pip install PyQt5
#   	* python -m pip install opencv-contrib-python
#   	* python -m pip install numpy
# 		* python -m pip install keyboard
#   	* python -m pip install win32con
# --------------------------------------------------------------------------------------------------------------------------
from keras.models import load_model
from keras.preprocessing import image
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5 import QtGui
import numpy as np
import sys
import os
import cv2
import numpy as np
import winGuiAuto
import win32gui
import win32con
import keyboard
import random as rd 
import string as st

# --------------------------------------------------------------------------------------------------------------------------
#	INIT THE GLOBAL DATA
#	You need to initialize the height & width of the images
# --------------------------------------------------------------------------------------------------------------------------
image_height, image_width = 64, 64

# --------------------------------------------------------------------------------------------------------------------------
#	LOAD THE TRAINED MODEL
#	You need to load the previously loaded model with the information of the alphabeth in sign language
# --------------------------------------------------------------------------------------------------------------------------
classifier = load_model('trained_model.h5')

# --------------------------------------------------------------------------------------------------------------------------
#	FUNCTION imagesSearchInFile()
#   Searches each file ending with .png in SampleGestures dirrectory so that custom gesture could be passed to predictor()
#   function
# --------------------------------------------------------------------------------------------------------------------------
def imagesSearchInFile():
    files = []
    for file in os.listdir("SampleGestures"):
        if file.endswith(".png"):
            files.append(file)
    return files

# --------------------------------------------------------------------------------------------------------------------------
#	FUNCTION openTemplateImage()
#	Displays predefined gesture image at right window
# --------------------------------------------------------------------------------------------------------------------------
def openTemplateImage():
    cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
    image = cv2.imread('images/template.png')
    cv2.imshow("Template", image)
    cv2.setWindowProperty(
        "Template", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Template", 298, 430)
    cv2.moveWindow("Template", 1052, 214)

# --------------------------------------------------------------------------------------------------------------------------
# 	FUNCTION shutDownCamera()
#	Shut downs the opened camera sended as a parameter
# --------------------------------------------------------------------------------------------------------------------------
def shutDownCamera(cam):
    cam.release()
    cv2.destroyAllWindows()

# --------------------------------------------------------------------------------------------------------------------------
# 	FUNCTION getRandomLetter()
#	Return a random letter of the alphabeth
# --------------------------------------------------------------------------------------------------------------------------
def getRandomLetter():
    return rd.choice(st.ascii_letters).upper()

# --------------------------------------------------------------------------------------------------------------------------
#	 FUNCTION prediction()
#	 Returns the predicted letter after comparing the read image with the prediction register with SIFT
# --------------------------------------------------------------------------------------------------------------------------
def prediction():
    test_image = image.load_img('images/prediction.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    results = classifier.predict(test_image)
    gesname = ''
    files = imagesSearchInFile()
    for i in range(len(files)):
        image_to_compare = cv2.imread("./SampleGestures/"+files[i])
        original = cv2.imread("images/prediction.png")
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio*n.distance:
                good_points.append(m)

        if(abs(len(good_points)+len(matches)) > 20):
            gesname = files[i]
            gesname = gesname.replace('.png', '')
            if(gesname == 'sp'):
                gesname = ' '
            return gesname

    if results[0][0] == 1:
        return 'A'
    elif results[0][1] == 1:
        return 'B'
    elif results[0][2] == 1:
        return 'C'
    elif results[0][3] == 1:
        return 'D'
    elif results[0][4] == 1:
        return 'E'
    elif results[0][5] == 1:
        return 'F'
    elif results[0][6] == 1:
        return 'G'
    elif results[0][7] == 1:
        return 'H'
    elif results[0][8] == 1:
        return 'I'
    elif results[0][9] == 1:
        return 'J'
    elif results[0][10] == 1:
        return 'K'
    elif results[0][11] == 1:
        return 'L'
    elif results[0][12] == 1:
        return 'M'
    elif results[0][13] == 1:
        return 'N'
    elif results[0][14] == 1:
        return 'O'
    elif results[0][15] == 1:
        return 'P'
    elif results[0][16] == 1:
        return 'Q'
    elif results[0][17] == 1:
        return 'R'
    elif results[0][18] == 1:
        return 'S'
    elif results[0][19] == 1:
        return 'T'
    elif results[0][20] == 1:
        return 'U'
    elif results[0][21] == 1:
        return 'V'
    elif results[0][22] == 1:
        return 'W'
    elif results[0][23] == 1:
        return 'X'
    elif results[0][24] == 1:
        return 'Y'
    elif results[0][25] == 1:
        return 'Z'

# --------------------------------------------------------------------------------------------------------------------------
#	CLASS Dashboard(QtWidgets.QMainWindow)
#	The Dashboard class is used to generate the main graphical interface from which the rest of the graphical interfaces 
# 	derive, the data and functionalities that are detailed in the .ui files are inserted
# --------------------------------------------------------------------------------------------------------------------------
class Dashboard(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Dashboard, self).__init__()
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowIcon(QtGui.QIcon('icons/windowLogo.png'))
        self.title = 'Aplicación para la enseñanza del abecedario en lenguaje de señas'
        self.mainMenu()

    def quitApplication(self):
        userReply = QMessageBox.question(
            self, 'Salir', "¿Quiere salir de la aplicación?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if userReply == QMessageBox.Yes:
            keyboard.press_and_release('alt+F4')
            
    def mainMenu(self):
        try:
            shutDownCamera(self.cam)
        except:
            pass
        uic.loadUi('UI_Files/dash.ui', self)
        self.setWindowTitle(self.title)
        self.scan_sinlge.clicked.connect(self.startScanner)
        self.exit_button.clicked.connect(self.quitApplication)
        self.scan_test.clicked.connect(self.startTest)
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_test.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.img_label.setPixmap(QPixmap('images/img_bg.jpg'))
        self._layout = self.layout()

    def startScanner(self):
        try:
            shutDownCamera(self.cam)
        except:
            pass
        uic.loadUi('UI_Files/scan_single.ui', self)
        self.setWindowTitle(self.title)
        self.pushButton_2.clicked.connect(lambda: shutDownCamera(self.cam))
        self.linkButton.clicked.connect(openTemplateImage)
        self.go_menu.clicked.connect(self.mainMenu)
        if self.scan_sinlge.clicked.connect(self.startScanner):
            self.cam = cv2.VideoCapture(0)
        self.scan_test.clicked.connect(self.startTest)
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_test.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.go_menu.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        img_text = ''
        
        while True:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            try:
                frame = cv2.resize(frame, (321, 270))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img1 = cv2.rectangle(
                    frame, (150, 50), (300, 200), (0, 255, 0), thickness=2, lineType=8, shift=0)
            except:
                keyboard.press_and_release('esc')

            height1, width1, channel1 = img1.shape
            step1 = channel1 * width1
            qImg1 = QImage(img1.data, width1, height1,
                           step1, QImage.Format_RGB888)

            try:
                self.label_3.setPixmap(QPixmap.fromImage(qImg1))
                slider1 = self.trackbar.value()
            except:
                pass

            lower_blue = np.array([0, 0, 0])
            upper_blue = np.array([179, 255, slider1])

            ROI = img1[52:198, 152:298]
            HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(HSV, lower_blue, upper_blue)

            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.imshow("mask", mask)
            cv2.setWindowProperty(
                "mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("mask", 118, 108)
            cv2.moveWindow("mask", 894, 271)

            hwnd = winGuiAuto.findTopWindow("mask")
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

            try:
                self.textBrowser.setText("\n\n\t"+str(img_text))
            except:
                pass

            img_name = "images/prediction.png"
            save_img = cv2.resize(mask, (image_height, image_width))
            cv2.imwrite(img_name, save_img)
            img_text = prediction()

            if cv2.waitKey(1) == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()
    
    def startTest(self):
        try:
            shutDownCamera(self.cam)
        except:
            pass
        uic.loadUi('UI_Files/scan_test.ui', self)
        self.setWindowTitle(self.title)
        self.pushButton_2.clicked.connect(lambda: shutDownCamera(self.cam))
        self.linkButton.clicked.connect(openTemplateImage)
        self.go_menu.clicked.connect(self.mainMenu)
        self.scan_sinlge.clicked.connect(self.startScanner)
        if self.scan_test.clicked.connect(self.startTest):
            self.cam = cv2.VideoCapture(0)
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_test.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.go_menu.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        img_text = ''
        random_letter = getRandomLetter()
        
        while True:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            try:
                frame = cv2.resize(frame, (321, 270))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img1 = cv2.rectangle(
                    frame, (150, 50), (300, 200), (0, 255, 0), thickness=2, lineType=8, shift=0)
            except:
                keyboard.press_and_release('esc')

            height1, width1, channel1 = img1.shape
            step1 = channel1 * width1
            qImg1 = QImage(img1.data, width1, height1,
                           step1, QImage.Format_RGB888)

            try:
                self.label_3.setPixmap(QPixmap.fromImage(qImg1))
                slider1 = self.trackbar.value()
            except:
                pass

            lower_blue = np.array([0, 0, 0])
            upper_blue = np.array([179, 255, slider1])

            ROI = img1[52:198, 152:298]
            HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(HSV, lower_blue, upper_blue)

            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.imshow("mask", mask)
            cv2.setWindowProperty(
                "mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("mask", 118, 108)
            cv2.moveWindow("mask", 894, 271)

            hwnd = winGuiAuto.findTopWindow("mask")
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

            try:
                self.textBrowser.setText(f"\n\n\n\t{random_letter} | "+str(img_text))
            except:
                pass

            img_name = "images/prediction.png"
            save_img = cv2.resize(mask, (image_height, image_width))
            cv2.imwrite(img_name, save_img)
            img_text = prediction()

            if cv2.waitKey(1) == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()

# --------------------------------------------------------------------------------------------------------------------------
#	 MAIN FUNCTION
# --------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	APP = QtWidgets.QApplication([])
	dashboard = Dashboard()
	dashboard.show()
	sys.exit(APP.exec())
