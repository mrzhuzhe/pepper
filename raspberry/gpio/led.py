# button 那一部分可以用软件输入

from gpiozero import LED
from time import sleep

led = LED(17)
while True:
    led.on()
    sleep(1)
    led.off()
    sleep(1)

