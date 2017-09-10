from kivy.app import App
from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, NoTransition
from kivy.graphics import Color, Ellipse, Line, Rectangle


Config.set('graphics', 'width', '1500')
Config.set('graphics', 'height', '800')

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import sys

class MainWindow(BoxLayout):
    bragg = ObjectProperty()
    graph = ObjectProperty()

    fps = ObjectProperty()
    time = NumericProperty(0)       # time in s (not real time)

    boxes = ListProperty()
    box_color = ListProperty()
    lines = ListProperty()
    line_color = ListProperty()

    d0 = ObjectProperty()
    d1 = ObjectProperty()
    d2 = ObjectProperty()
    n0 = ObjectProperty()
    n1 = ObjectProperty()
    n2 = ObjectProperty()

    fre = ObjectProperty()
    fre_min = NumericProperty(1)
    fre_max = NumericProperty(2)
    fre0 = NumericProperty(1)
    plt_intens = ListProperty()
    maxi = NumericProperty(0)

    def build(self):
        if self.fps.text != "":
            fps = float(self.fps.text)
        else:
            fps = 60
        Clock.schedule_interval(self.update, 1. / fps)

        # For plotting:
        #while True:
        #    self.update(0)

        x = np.arange(10)
        y = x**2

    def reschedule(self):
        Clock.unschedule(self.update)
        self.build()

    def update(self, dt):
        self.time += 0.1

        self.construct()

        self.bragg.canvas.clear()

        with self.bragg.canvas:
            for i, l_points in enumerate(self.lines):
                Color(*self.line_color[i])
                Line(points=l_points)
            for i, b_points in enumerate(self.boxes):
                Color(*self.box_color[i])
                Line(rectangle=b_points)

    @staticmethod
    def get_t(n1, n2):
        return 4 * n1 * n2 / (n1 + n2) ** 2

    @staticmethod
    def get_r(n1, n2):
        return ((n1 - n2) / (n1 + n2)) ** 2

    def construct(self):
        # Reset:
        self.lines = []
        self.line_color = []

        self.boxes = []
        self.box_color = []

        # Width of each material
        d0 = int(self.d0.text) if self.d0.text != "" else 100
        d1 = int(self.d1.text) if self.d1.text != "" else 100
        d2 = int(self.d2.text) if self.d2.text != "" else 100

        # Refractive index:
        n0 = float(self.n0.text) if self.n0.text != "" else 1
        n1 = float(self.n1.text) if self.n1.text != "" else 1
        n2 = float(self.n2.text) if self.n2.text != "" else 1

        # Intensity field
        I0_e = 100

        I1_t = I0_e * self.get_t(n0, n1)
        I2_t = I1_t * self.get_t(n1, n2)
        I3_t = I2_t * self.get_t(n2, n1)

        I0_r = I0_e * self.get_r(n0, n1)
        I1_r = I1_t * self.get_r(n1, n2)
        I2_r = I2_t * self.get_r(n2, n1)


        #self.fre0 += 0.001
        #print(self.fre0 - self.fre_min)
        #lam = self.fre0 / sc.c

        fre = self.fre.value
        lam = fre / sc.c


        w = 2 * sc.pi * sc.c / (lam * 1e-9)
        k = 2 * sc.pi / (lam * 1e-9)

        density = 0.1

        x0 = np.arange(0, d0, density)
        x1 = np.arange(0, d1, density)
        x2 = np.arange(0, d2, density)

        # Transmitted intensity ->
        y0_e = I0_e * np.real(np.exp(1.j * (-w * self.time + k * n0 * x0)))
        phi1 = k * n0 * d0
        self.lines.append(np.vstack((x0, y0_e + self.height / 2)).T.tolist())
        self.line_color.append((1, 0, 0, 1))
        # Second transmission ->
        y1_t = I1_t * np.real(np.exp(1.j * (-w * self.time + k * n1 * x1 + phi1)))
        phi2 = phi1 + k * n1 * d1
        self.lines.append(np.vstack((x1 + d0, y1_t + self.height / 2)).T.tolist())
        self.line_color.append((1, 0, 0, 1))
        # Third transmission ->
        y2_t = I2_t * np.real(np.exp(1.j * (-w * self.time + k * n2 * x2 + phi2)))
        phi3 = phi2 + k * n1 * d2
        self.lines.append(np.vstack((x2 + d0 + d1, y2_t + self.height / 2)).T.tolist())
        self.line_color.append((1, 0, 0, 1))
        # Fourth transmission ->
        y3_t = I3_t * np.real(np.exp(1.j * (-w * self.time + k * n1 * x1 + phi3)))
        self.lines.append(np.vstack((x1 + d0 + d1 + d2, y3_t + self.height / 2)).T.tolist())
        self.line_color.append((1, 0, 0, 1))

        # Reflected Material ->
        # Third reflection ->
        jump2 = 0 if n2 > n1 else np.pi
        y2_r = I2_r * np.real(np.exp(1.j * (-w * self.time + k * n2 * x2[::-1] + jump2 + phi3)))
        phi3_2 = phi3 + k * n2 * d2
        self.lines.append(np.vstack((x2 + d0 + d1, y2_r + self.height / 2)).T.tolist())
        self.line_color.append((0, 1, 0, 1))
        # Second reflection ->
        jump1 = 0 if n1 > n2 else np.pi
        y1_r = I1_r * np.real(np.exp(1.j * (-w * self.time + k * n1 * x1[::-1] + jump1 + phi2)))
        y2_r_t = I2_r * self.get_t(n2, n1) * np.real(np.exp(1.j * (-w * self.time + k * n1 * x1[::-1] + jump2 + phi3_2)))
        phi2_1 = phi2 + k * n1 * d1
        phi3_1 = phi3_2 + k * n1 * d1
        y1_r_sum = y1_r + y2_r_t
        self.lines.append(np.vstack((x1 + d0, y1_r_sum + self.height / 2)).T.tolist())
        self.line_color.append((0, 1, 0, 1))
        # First reflection ->
        jump0 = 0 if n0 > n1 else np.pi
        y0_r = I0_r * np.real(np.exp(1.j * (-w * self.time + k * n0 * x0[::-1] + jump0 + phi1)))
        y1_r_t = I1_r * self.get_t(n1, n0) * np.real(np.exp(1.j * (-w * self.time + k * n0 * x0[::-1] + jump1 + phi2_1)))
        y2_r_t = I2_r * self.get_t(n2, n1) * self.get_t(n1, n0) * np.real(np.exp(1.j * (-w * self.time + k * n0 * x0[::-1] + jump2 + phi3_1)))
        y0_r_sum = y0_r + y1_r_t + y2_r_t
        self.lines.append(np.vstack((x0, y0_r_sum + self.height / 2)).T.tolist())
        self.line_color.append((0, 1, 0, 1))
        self.maxi = float(max(y0_r_sum**2)**.5)

        """
        self.plt_intens.append((lam, self.maxi))
        if self.fre0 >= self.fre_max:
            print("Show")
            plt.plot(np.array(self.plt_intens).T[0], np.array(self.plt_intens).T[1])
            self.plt_intens = []
            plt.show()
            sys.exit()
        """

        # Place Material of different refractive index
        height = 200
        x = 0
        y = self.bragg.height / 2 - height / 2
        self.boxes.append((x, y, x + d0, y + height))
        self.box_color.append((0, 0, 1, 1))
        x += d0
        self.boxes.append((x, y, d1, y + height))
        self.box_color.append((0, 1, 0, 1))
        x += d1
        self.boxes.append((x, y, d2, y + height))
        self.box_color.append((1, 0, 0, 1))
        x += d2
        self.boxes.append((x, y, d1, y + height))
        self.box_color.append((0, 1, 0, 1))
        x += d1


class BraggApp(App):
    def build(self):
        m = MainWindow()
        m.build()
        return m


if __name__ == "__main__":
    BraggApp().run()
