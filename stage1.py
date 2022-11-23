import pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from pyqtgraph.Qt import QtCore, QtGui
from datetime import datetime
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import math
import numpy as np
import bisect
import cv2
import csv
import sys
from coordinates import ToGlobal, ToInternal


pencolour = 'b'
imgroot = 'map.png'


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
w = pg.GraphicsWindow()
w.setWindowTitle('Draggable')

p3 = w.addLayout(row=2, col=1)

proxy = QGraphicsProxyWidget()
SaveBut = QPushButton('Сохранить маршрут')
proxy.setWidget(SaveBut)
p3.addItem(proxy, row=1, col=3)

proxy2 = QGraphicsProxyWidget()
LoadBut = QPushButton('Загрузить маршрут')
proxy2.setWidget(LoadBut)
p3.addItem(proxy2, row=1, col=1)

proxy3 = QGraphicsProxyWidget()
AddPointBut = QPushButton('Добавить точку')
proxy3.setWidget(AddPointBut)
p3.addItem(proxy3, row=1, col=2)


def SavingEvent():
    global g
    filename = datetime.now().strftime("%d-%m-%y %H.%M.%S") + '.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['XAxis', 'YAxis']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in g.data['pos']:
            row = ToGlobal(row)
            writer.writerow({'XAxis': row[0], 'YAxis': row[1]})

SaveBut.clicked.connect(lambda: SavingEvent())

def get_file():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return str(filename)

def LoadingEvent():
    global g
    with open(get_file(), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        x, y = [], []
        for row in reader:
            temp = [float(row['XAxis']), float(row['YAxis'])]
            temp = ToInternal(temp)
            x.append(temp[0])
            y.append(temp[1])
    pos = np.column_stack((x, y))
    g.setData(pos=pos, size=10, pxMode=True, pen=None)

LoadBut.clicked.connect(lambda: LoadingEvent())


def AddPointEvent():
    global g
    pos = g.data['pos']
    cords = pos[-1]
    pos = np.append(pos, [[cords[0] - 5, cords[1]]], axis=0)
    g.setData(pos=pos, size=10, pxMode=True, pen=None)


AddPointBut.clicked.connect(lambda: AddPointEvent())


class Spline:

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        u"""
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
                 self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        u"""
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        u"""
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        u"""
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        u"""
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


class Spline2D:
    u"""
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        u"""
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw

    def calcd(self, s):
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        return np.array([dx, dy])

    def calcdd(self, s):
        ddx = self.sx.calcdd(s)
        ddy = self.sy.calcdd(s)
        return np.array([ddx, ddy])


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            if 'pos' in self.data:
                x, y = [], []
                for a in self.data['pos']:
                    x.append(a[0])
                    y.append(a[1])

                sp = Spline2D(x, y)
                s = np.arange(0, sp.s[-1], 0.1)
                rx, ry, ryaw, rk = [], [], [], []
                for i_s in s:
                    ix, iy = sp.calc_position(i_s)
                    rx.append(ix)
                    ry.append(iy)

                i = np.column_stack((rx, ry))
                self.data['posspline'] = np.array(i)

            self.data['adj'] = np.column_stack(
                (np.arange(0, npts - 1), np.arange(1, npts))
            )
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        if 'posspline' in self.data:
            v.clearPlots()
            x, y = [], []
            for i in self.data['posspline']:
                x.append(i[0])
                y.append(i[1])
            v.plot(x, y, pen=pencolour)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind][1] - pos[1]
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
        self.data['pos'][ind][0] = ev.pos()[0] + self.dragOffset
        self.updateGraph()
        if 'pos' in self.data:
            x, y = [], []
            for a in self.data['pos']:
                x.append(a[0])
                y.append(a[1])

            sp = Spline2D(x, y)
            s = np.arange(0, sp.s[-1], 0.1)
            rx, ry, ryaw, rk = [], [], [], []
            for i_s in s:
                ix, iy = sp.calc_position(i_s)
                rx.append(ix)
                ry.append(iy)

            i = np.column_stack((rx, ry))
            self.data['posspline'] = np.array(i)
        ev.accept()


g = Graph()
v = w.addPlot()
tr = QtGui.QTransform()  # prepare ImageItem transformation:
tr.scale(1, 1)  # scale horizontal and vertical axes

image = cv2.imread(imgroot, 1)
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
img = pg.ImageItem(image)
img.setPxMode(False)
tr.translate(-1 * (img.width()) / 2.0, -1 * img.height() / 2.0)  # move 3x3 image to locate center at axis origin
img.setTransform(tr)  # assign transform
v.addItem(img)
w.resize(img.width(), img.height())
v.vb.setLimits(xMin=-960, xMax=960, yMin=-540, yMax=540)
v.setAspectLocked()

v.addItem(g)
x = [-70.0, -68.5, -60.0, -50.0, -40.0, -25.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 25.0, 40.0, 50.0, 70.0, 50.0,
     25.0, 0.0, -5.0, -10.0, -25.0, -40.0, -50.0, -60.0, -68.5, -70.0]
y = [0.0, -10.0, -20.0, -24.29, -24.9, -20.26, -17.36, -9.25, -5.0, 0.0, 5.0, 10.0, 17.36, 20.26, 24.9, 24.29, 0.0,
     -24.29, -20.26, 0.0, 5.0, 9.25, 20.26, 24.9, 24.29, 20.0, 10.0, 0.0]
pos = np.column_stack((x, y))
g.setData(pos=pos, size=10, pxMode=True, pen=None)

x, y = [], []
for i in g.data['posspline']:
    x.append(i[0])
    y.append(i[1])
v.plot(x, y)

if __name__ == '__main__':

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
