# troubleShooting

1. 报错 from pyglet.gl import *
解决方案：pip3 install pyglet==1.5.11
虽然还是有：
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gym 0.18.0 requires pyglet<=1.5.0,>=1.4.0, but you have pyglet 1.5.11 which is incompatible.
这个报错 但是gym 可以使用