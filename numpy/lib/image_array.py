import numpy as np

__all__ = ['image_array']

class image_array:
    """
    image_array(pixel_array)

    Returns an object that takes in a pixel_array and provides
    methods to access and modify the pixel values as well as more complex
    image processing operations, such as adjusting brightness, contrast
    saturation, and converting the image to grayscale.

    Useful for image editing tasks.

    Parameters
    ----------
    pixel_array : array_like
        A 2D array of pixel values.
        Pixel values are expected to be iterables of length 3 or 4,
        representing RGB or RGBA values.
        RGB and RGBA values are expected to be in the range [0, 255].

    Attributes
    ----------
    pixel_array : ndarray
        The pixel_array that was passed in.

    Examples
    --------
    >>> import numpy as np

    >>> pixel_array = np.image_array([[(255, 0, 0), 
                        (0, 255, 0), (0, 0, 255)]])

    >>> pixel_array.get_pixel(0, 0)
    (255, 0, 0)

    >>> pixel_array.set_pixel(0, 0, (0, 0, 255))

    >>> pixel_array.get_dimensions()
    (3, 1)
    
    >>> pixel_array.grayscale()
    [[
        (76, 76, 76), 
        (149, 149, 149), 
        (29, 29, 29)
    ]]

    Brightness factor from 'adjust_brightness' method is a float. 
    A factor of 0 will make the image completely black,
    and a factor of 1 will make the image unchanged.
    
    >>> pixel_array.adjust_brightness(0.5)
    [[
        (127, 0, 0), 
        (0, 127, 0), 
        (0, 0, 127)
    ]]
    """
    def __init__(self, pixel_array):
        self.pixel_array = np.array(pixel_array)

    def get_pixel(self, x, y):
        return self.pixel_array[x][y]

    def set_pixel(self, x, y, value):
        self.pixel_array[x][y] = value

    def get_dimensions(self):
        height = len(self.pixel_array)
        if height > 0:
            width = len(self.pixel_array[0])
        else:
            width = 0
        return (width, height)

    def grayscale(self):
        for i in range(len(self.pixel_array)):
            for j in range(len(self.pixel_array[i])):
                pixel = self.pixel_array[i][j]
                gray = 0.299 * pixel[0] + 0.587 \
                        * pixel[1] + 0.114 * pixel[2]
                if len(pixel) == 4:
                    self.pixel_array[i][j] = (gray, gray, gray, pixel[3])
                else:
                    self.pixel_array[i][j] = (gray, gray, gray)
        return self.pixel_array
    

    def adjust_brightness(self, factor):
        if factor < 0:
            factor = 0
        for i in range(len(self.pixel_array)):
            for j in range(len(self.pixel_array[i])):
                pixel = self.pixel_array[i][j]
                r = max(0, min(255, pixel[0] * factor))
                g = max(0, min(255, pixel[1] * factor))
                b = max(0, min(255, pixel[2] * factor))
                new_pixel = (r, g, b)
                if len(pixel) == 4:
                    new_pixel = new_pixel + (pixel[3],)
                self.pixel_array[i][j] = new_pixel
        return self.pixel_array
    
    def rgb_to_hsv(self, pixel):
        r, g, b = pixel
        r, g, b = r / 255, g / 255, b / 255
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin
        if delta == 0:
            h = 0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * ((b - r) / delta + 2)
        elif cmax == b:
            h = 60 * ((r - g) / delta + 4)
        if cmax == 0:
            s = 0
        else:
            s = delta / cmax
        v = cmax
        return h, round(s, 3), round(v, 3)

    def hsv_to_rgb(self, pixel):
        h, s, v = pixel
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
        return round(r), round(g), round(b)
    
    def adjust_saturation(self, factor):
        for i in range(len(self.pixel_array)):
            for j in range(len(self.pixel_array[i])):
                pixel = self.pixel_array[i][j]
                h, s, v = self.rgb_to_hsv((pixel[0], pixel[1], pixel[2]))
                s = max(0, min(1, s * factor))
                new_pixel = self.hsv_to_rgb((h, s, v))
                if len(pixel) == 4:
                    new_pixel = new_pixel + (pixel[3],)
                self.pixel_array[i][j] = new_pixel
        return self.pixel_array
        

    def adjust_contrast(self, factor, midpoint = 128):
        factor = max(0, min(1, factor))
        for i in range(len(self.pixel_array)):
            for j in range(len(self.pixel_array[i])):
                pixel = self.pixel_array[i][j]
                r = midpoint + factor * (pixel[0] - midpoint)
                g = midpoint + factor * (pixel[1] - midpoint)
                b = midpoint + factor * (pixel[2] - midpoint)
                new_pixel = (r, g, b)
                if len(pixel) == 4:
                    new_pixel = new_pixel + (pixel[3],)
                self.pixel_array[i][j] = new_pixel
        return self.pixel_array