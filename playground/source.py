from playground.color import Color
from playground.shapes import Stroke, Circle
from playground.geometry import Vector

circles = []
for i in range(40):
    circles.append(Circle((i*50, 0), radius=i*2,
        fill='orange', opacity=70))

circles_2 = []
for i in range(32):
    circles_2.append(Circle((randint(0,2000), randint(0,1000)), radius=randint(10,100), fill=None, stroke=('white', 2)))

time = 0
def update(dt):
    global time
    time += dt
    for i, ci in enumerate(circles):
        ci.center.y = sin(time*1 + i/9) * 300
        ci.fill = Color(h=3*i+b+time*20)

    for i, ci in enumerate(circles_2):
        ci.radius += 1
        ci.stroke.fill.lightness = 130 / 2 - ci.radius / 2
        if ci.radius > 130:
            ci.center = Vector(randint(0,2000), randint(0,1000))
            ci.radius = 30
	