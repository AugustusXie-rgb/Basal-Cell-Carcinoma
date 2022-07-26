# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QApplication
import os

from run_IG_UI import run_IG_UI
from Grad_CAM_UI import Grad_CAM_UI
from run_test_UI import run_test_UI
from tif2bmp import tif2bmp                         # update 2021-07-02
from quality_indicator import quality_indicator     # update 2021-07-02


class Ui_MainWindow(object):
    def __init__(self):
        self.imgList = []
        self.folderPath = '/home/xingpeng/Project/Interface_Demo/UI_work/sample_images_bmp/'  # by default
        self.modelPath = '/home/xingpeng/Project/Interface_Demo/UI_work/checkpoint-95e-val_accuracy_0.93.hdf5'
        self.outputFolderPath = './output/'
        self.imgNum = 0
        self.imgIdx = 0
        self.pix = QPixmap()

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screen_height = self.screenRect.height()  # 1080
        self.screen_width = self.screenRect.width()  # 1920

        self.test_run = False
        self.has_model = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        MainWindow.resize(self.screen_width, self.screen_height)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.scene = QtWidgets.QGraphicsScene(self.centralwidget)

        self.folderSelectButton = QtWidgets.QPushButton(self.centralwidget)
        self.folderSelectButton.setGeometry(QtCore.QRect(10, 10, 90, 30))
        self.folderSelectButton.setObjectName("folderSelectButton")
        self.folderSelectButton.clicked.connect(self.openFolderClicked)

        self.folderPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.folderPathLabel.setGeometry(QtCore.QRect(110, 10, 600, 30))
        self.folderPathLabel.setObjectName("folderPathLabel")
        self.folderPathLabel.setText("Image Folder Path")

        self.modelSelectButton = QtWidgets.QPushButton(self.centralwidget)
        self.modelSelectButton.setGeometry(QtCore.QRect(780, 10, 90, 30))
        self.modelSelectButton.setObjectName("modelSelectButton")
        self.modelSelectButton.setText("Load Model")
        self.modelSelectButton.clicked.connect(self.modelSelectClicked)

        self.modelPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.modelPathLabel.setGeometry(QtCore.QRect(880, 10, 600, 30))
        self.modelPathLabel.setObjectName("modelPathLabel")
        self.modelPathLabel.setText("Model Path")

        self.rawImageView = QtWidgets.QGraphicsView(self.centralwidget)
        self.rawImageView.setGeometry(QtCore.QRect(10, 50, 700, 700))
        self.rawImageView.setObjectName("rawImageView")

        self.horizontalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(10, 760, 700, 15))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")

        self.prevImgButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevImgButton.setGeometry(QtCore.QRect(10, 785, 90, 20))
        self.prevImgButton.setObjectName("prevImgButton")
        self.prevImgButton.setText('<-')
        self.prevImgButton.clicked.connect(self.prevImgClicked)

        self.nextImgButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextImgButton.setGeometry(QtCore.QRect(620, 785, 90, 20))
        self.nextImgButton.setObjectName("nextImgButton")
        self.nextImgButton.setText('->')
        self.nextImgButton.clicked.connect(self.nextImgClicked)

        self.gradCamView = QtWidgets.QGraphicsView(self.centralwidget)
        self.gradCamView.setGeometry(QtCore.QRect(780, 50, 500, 500))
        self.gradCamView.setObjectName("gradCamView")

        self.IGView = QtWidgets.QGraphicsView(self.centralwidget)
        self.IGView.setGeometry(QtCore.QRect(1350, 50, 500, 500))
        self.IGView.setObjectName("IGView")

        self.arrayValueLabel = QtWidgets.QLabel(self.centralwidget)
        self.arrayValueLabel.setGeometry(QtCore.QRect(780, 600, 500, 100))
        self.arrayValueLabel.setObjectName("arrayValueLabel")
        self.arrayValueLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.arrayValueLabel.setText("array value")

        self.singleValueLabel = QtWidgets.QLabel(self.centralwidget)
        self.singleValueLabel.setGeometry(QtCore.QRect(1350, 600, 500, 100))
        self.singleValueLabel.setObjectName("singleValueLabel")
        self.singleValueLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.singleValueLabel.setText("single value")

        self.qualityIndicatorLabel = QtWidgets.QLabel(self.centralwidget)           # update 2021-07-02
        self.qualityIndicatorLabel.setGeometry(QtCore.QRect(780, 800, 500, 100))    # update 2021-07-02
        self.qualityIndicatorLabel.setObjectName("qualityIndicatorLabel")           # update 2021-07-02
        self.qualityIndicatorLabel.setAlignment(QtCore.Qt.AlignCenter)              # update 2021-07-02
        self.qualityIndicatorLabel.setText("quality indicator")                     # update 2021-07-02

        self.runButton = QtWidgets.QPushButton(self.centralwidget)
        self.runButton.setGeometry(QtCore.QRect(315, 785, 90, 40))
        self.runButton.setObjectName("runButton")
        self.runButton.clicked.connect(self.runDemoClicked)

        self.statusInfo = QtWidgets.QLabel(self.centralwidget)
        self.statusInfo.setGeometry(QtCore.QRect(60, 835, 600, 30))
        self.statusInfo.setObjectName("statusInfo")
        self.statusInfo.setAlignment(QtCore.Qt.AlignCenter)
        self.statusInfo.setText("Status")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demo"))
        self.folderSelectButton.setText(_translate("MainWindow", "Open Folder"))
        self.folderPathLabel.setText(_translate("MainWindow", "TextLabel"))
        self.runButton.setText(_translate("MainWindow", "Run"))

    def openFolderClicked(self):
        self.folderPath = QFileDialog.getExistingDirectory(self.centralwidget, 'select a folder', os.getcwd())
        self.folderPathLabel.setText(self.folderPath)
        #self.folderPath = '/home/xingpeng/Project/Interface_Demo/UI_work/sample_images_bmp/'

        self.folderPath = tif2bmp(self.folderPath)      # update 2021-07-02

        self.imgList = sorted(os.listdir(self.folderPath))
        self.imgNum = len(self.imgList)

        self.showImage()

    def modelSelectClicked(self):
        self.modelPath = QFileDialog.getOpenFileName(self.centralwidget, 'select a model file (.hdf5)', os.getcwd(), 'Model files (*.hdf5)')
        self.modelPathLabel.setText(self.modelPath[0])
        self.statusInfo.setText('Model loaded')
        self.has_model = True

    def prevImgClicked(self):
        if self.imgIdx > 0:
            self.imgIdx -= 1
        else:
            return

        self.showImage()

    def nextImgClicked(self):
        if self.imgIdx < self.imgNum-1:
            self.imgIdx += 1
        else:
            return

        self.showImage()

    def runDemoClicked(self):
        if self.has_model is False:
            self.statusInfo.setText('No model loaded')
            return

        self.statusInfo.setText('Running IG ...')
        run_IG_UI(model_location=self.modelPath[0], img_folder=self.folderPath, output_folder=self.outputFolderPath)

        self.statusInfo.setText('Running Grad CAM ...')
        Grad_CAM_UI(model_location=self.modelPath[0], img_folder=self.folderPath, output_folder=self.outputFolderPath)

        self.statusInfo.setText('Running test ...')
        v_array, v_number = run_test_UI(model_location=self.modelPath[0], img_folder=self.folderPath)

        v_array = str(v_array)
        v_array = v_array[1:-1]
        v_number = str(v_number)
        self.arrayValueLabel.setText(v_array)
        self.singleValueLabel.setText(v_number)

        self.test_run = True
        self.showImage()

    def showImage(self):
        img_name = self.imgList[self.imgIdx]
        img_path = self.folderPath + img_name
        qi = quality_indicator(img_path)                # update 2021-07-02
        self.qualityIndicatorLabel.setText(str(qi))     # update 2021-07-02

        self.pix.load(img_path)
        self.pix = self.pix.scaledToHeight(695)
        item = QtWidgets.QGraphicsPixmapItem(self.pix)
        self.scene.addItem(item)
        self.rawImageView.setScene(self.scene)

        self.folderPathLabel.setText(img_path)

        if self.test_run:
            IG_img_path = self.outputFolderPath + 'IG_' + img_name
            self.pix.load(IG_img_path)
            self.pix = self.pix.scaledToHeight(495)
            item = QtWidgets.QGraphicsPixmapItem(self.pix)
            self.scene.addItem(item)
            self.IGView.setScene(self.scene)

            GradCAM_img_path = self.outputFolderPath + 'Grad_' + img_name
            self.pix.load(GradCAM_img_path)
            self.pix = self.pix.scaledToHeight(495)
            item = QtWidgets.QGraphicsPixmapItem(self.pix)
            self.scene.addItem(item)
            self.gradCamView.setScene(self.scene)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
