# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\_files\VSCode\TIK\lab2\EntropyCalculatorFromCounts.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLineEdit

from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp

import math;
import numpy as np
import entropy as ntrp


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(243, 136)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        Dialog.setFont(font)

        class DigitDotSemicolonInput(QLineEdit):
            def __init__(self, parent=None):
                super().__init__(parent)
                regex = QRegExp(r"[0-9;]*")
                validator = QRegExpValidator(regex, self)
                self.setValidator(validator)
        
        self.lineEdit_insert_A = DigitDotSemicolonInput(Dialog)
        self.lineEdit_insert_A.setEnabled(True)
        self.lineEdit_insert_A.setGeometry(QtCore.QRect(70, 30, 113, 20))
        self.lineEdit_insert_A.setObjectName("lineEdit_insert")

        self.label_insert = QtWidgets.QLabel(Dialog)
        self.label_insert.setGeometry(QtCore.QRect(40, 10, 171, 20))
        self.label_insert.setAlignment(QtCore.Qt.AlignCenter)
        self.label_insert.setObjectName("label_insert")
        
        self.pushButton_Calculate = QtWidgets.QPushButton(Dialog)
        self.pushButton_Calculate.setGeometry(QtCore.QRect(90, 60, 75, 23))
        self.pushButton_Calculate.setObjectName("pushButton_Calculate")
        self.pushButton_Calculate.clicked.connect(self.calc)

        self.lineEdit_result = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_result.setEnabled(False)
        self.lineEdit_result.setGeometry(QtCore.QRect(70, 100, 113, 20))
        self.lineEdit_result.setObjectName("lineEdit_result")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Підраховувач ентропії"))
        self.label_insert.setText(_translate("Dialog", "Кількості повідомлень"))
        self.pushButton_Calculate.setText(_translate("Dialog", "Вирахувати"))

    def calc(self):
        input_str = self.lineEdit_insert_A.text()
        input_arr = np.array(input_str.split(";")).astype(int)

        entropy = ntrp.unconditional_entropy_from_array(input_arr)
        self.lineEdit_result.setText(f"{entropy}")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
