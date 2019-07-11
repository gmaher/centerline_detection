import numpy as np

class Shape(object):
    def __init__(self, box, rgb):
        self.box = box
        self.rgb = rgb

    def paint(self, x, H, W):
        raise RuntimeError("abstract not implemented")

class Square(Shape):
    def paint(self, x, W, H):
        ox = int(self.box[0]*(W-1))
        oy = int(self.box[1]*(H-1))

        w = int(self.box[2]*(W-1))
        h = int(self.box[3]*(H-1))

        for i,c in enumerate(self.rgb):
            x[oy:oy+h,ox:ox+w,i] = c

class Circle(Shape):
    def paint(self, x, W, H):
        ox = int(self.box[0]*(W-1))
        oy = int(self.box[1]*(H-1))

        w = int(self.box[2]*(W-1))
        h = int(self.box[3]*(H-1))

        r = h/2

        cx = ox+r
        cy = oy+r

        for i in range(oy,oy+h):
            for j in range(ox,ox+w):
                if (cy-i)**2+(cx-j)**2 < r**2:
                    x[i,j,:] = self.rgb

class Triangle(Shape):
    def paint(self, x, W, H):
        ox = int(self.box[0]*(W-1))
        oy = int(self.box[1]*(H-1))

        w = int(self.box[2]*(W-1))
        h = int(self.box[3]*(H-1))

        for i,c in enumerate(self.rgb):
            for j in range(0,h):
                x[oy+h-j, ox+j:ox+w-j,i] = c
