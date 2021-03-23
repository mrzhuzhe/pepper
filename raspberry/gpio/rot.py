# https://gpiozero.readthedocs.io/en/stable/recipes.html?highlight=RotaryEncoder#rotary-encoder

#   from threading import Event
#   from colorzero import Color
from gpiozero import RotaryEncoder, RGBLED, Button

rotor = RotaryEncoder(16, 20, wrap=True, max_steps=7)
rotor.steps = -7
#   led = RGBLED(22, 23, 24, active_high=False)
btn = Button(21)
#btn = Button(21, pull_up=False)
#   led.color = Color('#f00')
#   done = Event()

def change_hue():
    # Scale the rotor steps (-180..180) to 0..1
    hue = (rotor.steps + 7) / 14
    print(rotor.steps)
    #led.color = Color(h=hue, s=1, v=1)

def show_color():
        print('btn release')
#    print('Hue {led.color.hue.deg:.1f}° = {led.color.html}'.format(led=led))

def stop_script():
    print('btn hold')
#    done.set()

print('Select a color by turning the knob')
while True:
    rotor.when_rotated = change_hue
#print('Push the button to see the HTML code for the color')
    btn.when_pressed = show_color
#    btn.when_released = show_color
#print('Hold the button to exit')
    btn.when_held = stop_script
#done.wait()