from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp

hands_class = mp.solutions.hands
hands = hands_class.Hands()
mpDraw = mp.solutions.drawing_utils

Builder.load_string("""
#:kivy 2.0.0

<MenuScreen>:
    MainWidget:


<SettingsScreen>:
    BoxLayout:  
        orientation: 'vertical'
        Label:
            text: "Reconocimiento gestual"
        Cam:
            id: cam1
            
        Button:
            text: 'Volver al menu'
            on_press: root.manager.current = 'menu'
""")


class MainWidget(Widget):

    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        btnnext = Button(text='Ir a la App', pos=(200, 400))
        btnnext.bind(on_press=self.gonext)
        self.add_widget(btnnext)

    def gonext(self, btn_inst):
        sm.current = "settings"


class MenuScreen(Screen):
    pass


class SettingsScreen(Screen):
    pass


class Cam(Image):

    def on_kv_post(self, base_widget):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(
                    frame, hand, hands_class.HAND_CONNECTIONS)

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture1


class ScreenManager(ScreenManager):
    pass


sm = ScreenManager()
sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(SettingsScreen(name='settings'))


class TestApp(App):

    def build(self):
        return sm


if __name__ == '__main__':
    TestApp().run()
