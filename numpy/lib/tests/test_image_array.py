import numpy as np
from numpy.lib import image_array
from numpy.testing import assert_array_equal

class TestImageArray:
    def test_get_pixel(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        assert np.all(image.get_pixel(0, 0) == (255, 0, 0))
        assert np.all(image.get_pixel(0, 1) == (0, 255, 0))
        assert np.all(image.get_pixel(0, 2) == (0, 0, 255))

    def test_set_pixel(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        image.set_pixel(0, 0, (0, 0, 255))
        assert np.all(image.get_pixel(0, 0) == (0, 0, 255))

    def test_get_dimensions(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        assert image.get_dimensions() == (3, 1)

    def test_get_dimensions_empty(self):
        pixel_array = [[]]
        image = image_array(pixel_array)
        assert image.get_dimensions() == (0, 1)

    def test_get_dimensions_empty_2(self):
        pixel_array = []
        image = image_array(pixel_array)
        assert image.get_dimensions() == (0, 0)

    def test_multiple_rows(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)], 
                       [(0, 255, 0), (0, 0, 255), (255, 0, 0)], 
                       [(0, 0, 255), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        assert image.get_dimensions() == (3, 3)
        assert np.all(image.get_pixel(0, 0) == (255, 0, 0))
        assert np.all(image.get_pixel(0, 2) == (0, 0, 255))
        assert np.all(image.get_pixel(1, 0) == (0, 255, 0))
        assert np.all(image.get_pixel(1, 2) == (255, 0, 0))
        assert np.all(image.get_pixel(2, 0) == (0, 0, 255))
        assert np.all(image.get_pixel(2, 2) == (0, 0, 255))

        image.set_pixel(2, 0, (255, 0, 0))
        assert np.all(image.get_pixel(2, 0) == (255, 0, 0))


    def test_grayscale(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        grayscale_array = image.grayscale()
        expected_array = [[(76, 76, 76), 
                                    (149, 149, 149), 
                                    (29, 29, 29)]]
        assert_array_equal(grayscale_array, expected_array)

    def test_grayscale_empty(self):
        pixel_array = [[]]
        image = image_array(pixel_array)
        grayscale_array = image.grayscale()
        expected_array = [[]]
        assert_array_equal(grayscale_array, expected_array)

    def test_grayscale_with_alpha(self):
        pixel_array = [[(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]]
        image = image_array(pixel_array)
        grayscale_array = image.grayscale()
        expected_array = [[(76, 76, 76, 255), 
                                    (149, 149, 149, 255), 
                                    (29, 29, 29, 255)]]
        assert_array_equal(grayscale_array, expected_array)

    def test_adjust_brightness(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        brightness_array = image.adjust_brightness(0.5)
        expected_array = [[(127, 0, 0), 
                                    (0, 127, 0), 
                                     (0, 0, 127)]]
        assert_array_equal(brightness_array, expected_array)

    def test_adjust_brightness_empty(self):
        pixel_array = [[]]
        image = image_array(pixel_array)
        brightness_array = image.adjust_brightness(0.5)
        expected_array = [[]]
        assert_array_equal(brightness_array, expected_array)
    
    def test_adjust_brightness_with_alpha(self):
        pixel_array = [[(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]]
        image = image_array(pixel_array)
        brightness_array = image.adjust_brightness(0.5)
        expected_array = [[(127, 0, 0, 255), 
                                    (0, 127, 0, 255), 
                                    (0, 0, 127, 255)]]
        assert_array_equal(brightness_array, expected_array)

    def test_adjust_brightness_over_max(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        brightness_array = image.adjust_brightness(2)
        expected_array = [[(255, 0, 0), 
                                    (0, 255, 0), 
                                    (0, 0, 255)]]
        assert_array_equal(brightness_array, expected_array)

    def test_adjust_brightness_under_min(self):
        pixel_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        image = image_array(pixel_array)
        brightness_array = image.adjust_brightness(-1)
        expected_array = [[(0, 0, 0), 
                                    (0, 0, 0), 
                                    (0, 0, 0)]]
        assert_array_equal(brightness_array, expected_array)

    def test_rgb_hsv(self):
        image = image_array([[]])
        pixel = (255, 0, 0)
        assert image.rgb_to_hsv(pixel) == (0, 1, 1)

        pixel = (0, 255, 0)
        assert image.rgb_to_hsv(pixel) == (120, 1, 1)

        pixel = (120, 200, 120)
        assert image.rgb_to_hsv(pixel) == \
            (120, 0.4, 0.784)

        pixel = (0, 0, 0)
        assert image.rgb_to_hsv(pixel) == (0, 0, 0)

        pixel = (255, 255, 255)
        assert image.rgb_to_hsv(pixel) == (0, 0, 1)


    def test_hsv_rgb(self):
        image = image_array([[]])
        pixel = (0, 1, 1)
        assert image.hsv_to_rgb(pixel) == (255, 0, 0)

        pixel = (120, 1, 1)
        assert image.hsv_to_rgb(pixel) == (0, 255, 0)

        pixel = (120, 0.4, 0.784)
        print(image.hsv_to_rgb(pixel))
        assert image.hsv_to_rgb(pixel) == (120, 200, 120)

        pixel = (0, 0, 0)
        assert image.hsv_to_rgb(pixel) == (0, 0, 0)

        pixel = (0, 0, 1)
        assert image.hsv_to_rgb(pixel) == (255, 255, 255)


    def test_adjust_saturation(self):
        pixel_array = image_array([[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                             [(0, 255, 0), (0, 0, 255), (255, 0, 0)],
                             [(0, 0, 255), (0, 255, 0), (0, 0, 255)]])
        saturation_array = pixel_array.adjust_saturation(2)
        expected_array = [[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                          [(0, 255, 0), (0, 0, 255), (255, 0, 0)],
                          [(0, 0, 255), (0, 255, 0), (0, 0, 255)]]
        
        assert_array_equal(saturation_array, expected_array)
        
        pixel_array = image_array([
                            [(120, 0, 200), (150, 0, 120), (0, 190, 0)], 
                            [(0, 0, 0), (120, 120, 120), (120, 200, 100)], 
                            [(255, 255, 255), (0, 0, 0), (255, 255, 120)]])
        
        saturation_array = pixel_array.adjust_saturation(0.5)
        expected_array = [[(160, 100, 200), (150, 75, 135), (95, 190, 95)], 
                          [(0, 0, 0), (120, 120, 120), (160, 200, 150)], 
                          [(255, 255, 255), (0, 0, 0), (255, 255, 188)]]
        
        assert_array_equal(saturation_array, expected_array)

    
    def test_adjust_saturation_empty(self):
        pixel_array = image_array([[]])
        saturation_array = pixel_array.adjust_saturation(2)
        expected_array = [[]]
        assert_array_equal(saturation_array, expected_array)
        
        pixel_array = image_array([])
        saturation_array = pixel_array.adjust_saturation(2)
        expected_array = []
        assert_array_equal(saturation_array, expected_array)
        

    def test_adjust_saturation_with_alpha(self):
        pixel_array = image_array([
                    [(255, 0, 0, 160), (0, 255, 0, 128), (0, 0, 255, 170)],
                     [(0, 255, 0, 0), (0, 0, 255, 5), (255, 0, 0, 255)],
                     [(0, 0, 255, 255), (0, 255, 0, 255), (0, 0, 255, 255)]])
        saturation_array = pixel_array.adjust_saturation(2)
        expected_array = [
                  [(255, 0, 0, 160), (0, 255, 0, 128), (0, 0, 255, 170)],
                  [(0, 255, 0, 0), (0, 0, 255, 5), (255, 0, 0, 255)],
                  [(0, 0, 255, 255), (0, 255, 0, 255), (0, 0, 255, 255)]]
        
        assert_array_equal(saturation_array, expected_array)
        
        pixel_array = image_array([
                [(120, 0, 200, 255), (150, 0, 120, 255), (0, 190, 0, 255)], 
                [(0, 0, 0, 255), (120, 120, 120, 255), (120, 200, 100, 255)], 
                [(255, 255, 255, 255), (0, 0, 0, 255), (255, 255, 120, 255)]])
        
        saturation_array = pixel_array.adjust_saturation(0.5)
        expected_array = [
            [(160, 100, 200, 255), (150, 75, 135, 255), (95, 190, 95, 255)], 
            [(0, 0, 0, 255), (120, 120, 120, 255), (160, 200, 150, 255)], 
            [(255, 255, 255, 255), (0, 0, 0, 255), (255, 255, 188, 255)]]
        
        assert_array_equal(saturation_array, expected_array)

    def test_adjust_contrast(self):
        pixel_array = image_array(
            [[(0, 120, 120), (0, 200, 100), (100, 200, 100)],
             [(150, 100, 100), (200, 200, 200), (50, 50, 50)],
             [(0, 0, 0), (255, 255, 255), (0, 0, 0)]])
        
        contrast_array = pixel_array.adjust_contrast(0.5)
        expected_array = [[(64, 124, 124), (64, 164, 114), (114, 164, 114)], 
                          [(139, 114, 114), (164, 164, 164), (89, 89, 89)], 
                          [(64, 64, 64), (191, 191, 191), (64, 64, 64)]]
        
        assert_array_equal(contrast_array, expected_array)

    def test_adjust_contrast_empty(self):
        pixel_array = image_array([[]])
        contrast_array = pixel_array.adjust_contrast(2)
        expected_array = [[]]
        assert_array_equal(contrast_array, expected_array)
        
        pixel_array = image_array([])
        contrast_array = pixel_array.adjust_contrast(2)
        expected_array = []
        assert_array_equal(contrast_array, expected_array)

    def test_adjust_contrast_with_alpha(self):
        pixel_array = image_array(
            [[(0, 120, 120, 255), (0, 200, 100, 255), (100, 200, 100, 255)],
             [(150, 100, 100, 255), (200, 200, 200, 255), (50, 50, 50, 255)],
             [(0, 0, 0, 255), (255, 255, 255, 255), (0, 0, 0, 255)]])
        
        contrast_array = pixel_array.adjust_contrast(0.5)
        expected_array = [
            [(64, 124, 124, 255), (64, 164, 114, 255), (114, 164, 114, 255)],
            [(139, 114, 114, 255), (164, 164, 164, 255), (89, 89, 89, 255)],
            [(64, 64, 64, 255), (191, 191, 191, 255), (64, 64, 64, 255)]]
        
        assert_array_equal(contrast_array, expected_array)

    def test_adjust_contrast_empty(self):
        pixel_array = image_array([[]])
        contrast_array = pixel_array.adjust_contrast(2)
        expected_array = [[]]
        assert_array_equal(contrast_array, expected_array)
        
        pixel_array = image_array([])
        contrast_array = pixel_array.adjust_contrast(2)
        expected_array = []
        assert_array_equal(contrast_array, expected_array)

        
        
