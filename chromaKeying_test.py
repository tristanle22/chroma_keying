import unittest
import numpy as np
import cv2
from chromaKeying import ChromaKeyer

settings_path = './chromaKeyingSetinggs.json'

class TestChromaKeyingMethods(unittest.TestCase):
  # def test_detect_background(self):
  #   keyer = ChromaKeyer(None,False,settings_path)
  #   all_black = np.zeros((10,10),dtype='uint8')
  #   all_white = np.ones((10,10),dtype='uint8')
  #   all_green = np.zeros((10,10,3),dtype='uint8')
  #   all_green[:,:,:] = (0,255,0)
  #   most_green = all_green
  #   most_green[:3,:3,:] = (255,0,0)

  #   keyer.set_video(all_black)
  #   self.assertEqual(keyer.key(),"Result")

  #   keyer.set_video(all_white)
  #   self.assertEqual(keyer.key(),"Result")

  #   keyer.set_video(all_green)
  #   self.assertEqual(keyer.key(),"Result")

  #   keyer.set_video(most_green)
  #   self.assertEqual(keyer.key(),"Result")


  def test_detect_foreground_absolute(self):
    pass

  def test_detect_foreground_reflective(self):
    pass

  def test_supress_color_spill(self):
    pass

  def test_lightness_grouping(self):
    width, height = 320, 320
    pixel_count = width * height
    keyer = ChromaKeyer(None, False, None)

    test_image_0 = np.zeros((width,height,3),dtype='float32')
    test_image_0[:width//2, :height//2, 2] = 0.15
    lightness_masks = keyer.lightness_grouping(test_image_0)
    self.assertEqual(pixel_count//4, np.sum(lightness_masks[1]) / 255)

    test_image_1 = np.zeros((width, height), dtype='float32')
    lightness_masks = keyer.lightness_grouping(test_image_1)
    self.assertEqual([], lightness_masks)

    test_image_2 = np.zeros((width, height, 3), dtype='float32')
    test_image_2[:width//2, : , 2] = 0.1
    lightness_masks = keyer.lightness_grouping(test_image_2)
    self.assertEqual(pixel_count/2, np.sum(lightness_masks[1]) / 255)

    test_image_rand = np.random.rand(height,width,3)
    lightness_masks = keyer.lightness_grouping(test_image_rand)
    self.assertEqual(pixel_count, np.sum(lightness_masks) / 255)

  def test_smoothen_histogrmam(self):
    test_image = cv2.imread("test_6.jpg")
    test_image_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    test_image_hsv = test_image_hsv.astype(np.float32)
    test_image_hsv /= 255.0

    hist = cv2.calcHist([test_image_hsv], [0,1], None, [256,256], [0.0, 1.0, 0.0, 1.0])
    keyer = ChromaKeyer(None, False, None)

    test_vars = []
    for x in range(5,10,2):
      a = keyer.smoothen_histogram(hist, gamma=x)
      var = self.var_abs_laplacian(a)
      test_vars.append(var)

    self.assertLess(test_vars[1][0], test_vars[0][0])
    self.assertLess(test_vars[2][0], test_vars[1][0])
    self.assertLess(test_vars[1][1], test_vars[0][1])
    self.assertLess(test_vars[2][1], test_vars[1][1])

  def var_abs_laplacian(self, image):
    if not image.any():
      print("Null image")
      return 0
    M,N = image.shape
    depth = cv2.CV_64F

    sobel_x = cv2.Sobel(image, depth, 1, 0)
    sobel_y = cv2.Sobel(image, depth, 0, 1)
    average_x = 1/(M*N) * np.sum(np.abs(sobel_x))
    average_y = 1/(M*N) * np.sum(np.abs(sobel_y))
    sobel_x_var = np.sum(np.square(sobel_x - average_x))
    sobel_y_var = np.sum(np.square(sobel_y - average_y))

    return (sobel_x_var, sobel_y_var)

if __name__ == '__main__':
    unittest.main()