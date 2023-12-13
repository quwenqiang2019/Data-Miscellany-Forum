class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def area(self):
        return self._width * self._height

r = Rectangle(3, 4)
print(r.area)  # 输出：12