import numpy as np

class Shape(object):
    def __init__(self, box, rgb):
        self.box = box
        self.rgb = rgb

    def paint(self, x, H, W):
        raise RuntimeError("abstract not implemented")

class Square(Shape):
    shape_type = 0
    def paint(self, x, W, H):
        ox = int(self.box[0]*(W-1))
        oy = int(self.box[1]*(H-1))

        w = int(self.box[2]*(W-1))
        h = int(self.box[3]*(H-1))

        for i,c in enumerate(self.rgb):
            x[oy:oy+h,ox:ox+w,i] = c

class Circle(Shape):
    shape_type = 1
    def __init__(self, box, rgb):
        self.box = box
        self.box[2] = self.box[3]
        self.rgb = rgb

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
                if (cy-i)**2+(cx-j)**2 <= r**2 and i < H and j < W:
                    x[i,j,:] = self.rgb

class Triangle(Shape):
    shape_type = 2
    def __init__(self, box, rgb):
        self.box = box
        self.box[2] = self.box[3]
        self.rgb = rgb
    def paint(self, x, W, H):
        ox = int(self.box[0]*(W-1))
        oy = int(self.box[1]*(H-1))

        w = int(self.box[2]*(W-1))
        h = int(self.box[3]*(H-1))

        for i,c in enumerate(self.rgb):
            for j in range(0,h):
                width = int((h-j)*1.0/h*w)
                d = int( (w-width)*1.0/2 )
                x[oy+h-j, ox+d:ox+d+width,i] = c

def get_random_shapes(n_min, n_max, H, W, shape_dist=[0.333,0.333,0.334]):
    n_shapes = np.random.randint(n_min,n_max)

    shapes = []
    x = np.zeros((H,W,3))

    types = [0,1,2]
    hmax = 1.0/n_shapes

    for i in range(n_shapes):
        oy = (1.0-i*1.0/n_shapes)-hmax/2
        ox = np.random.rand()*0.6+0.2
        h  = np.random.rand()*(hmax-0.1)+0.1
        w  = np.random.rand()*0.1+0.1
        if ox+w > 1:
            w = 0.99-ox
        if oy+h > 1:
            h = 0.99-oy

        box = np.array([ox,oy,w,h])

        rgb = np.random.randint(255, size=3)

        typ = np.random.choice(types, p=shape_dist)

        if typ == 0:
            s = Square(box,rgb)
        elif typ == 1:
            s = Circle(box,rgb)
        else:
            s = Triangle(box,rgb)

        shapes.append(s)
        s.paint(x, W, H)

    return x,shapes
