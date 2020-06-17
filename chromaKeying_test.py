import unittest
import numpy as np
import cv2
import chromaKeying

settings_path = 'chromaKeyingSetinggs.json'

class TestChromaKeyingMethods(unittest.TestCase):
  def test_detect_background(self):
    keyer = chromaKeying.ChromaKeyer(None,False,settings_path)
    all_black = np.zeros((10,10),dtype='uint8')
    all_white = np.ones((10,10),dtype='uint8')
    all_green = np.zeros((10,10,3),dtype='uint8')
    all_green[:,:,:] = (0,255,0)
    most_green = all_green
    most_green[:3,:3,:] = (255,0,0)

    keyer.set_video(all_black)
    self.assertEqual(keyer.key(),"Result")

    keyer.set_video(all_white)
    self.assertEqual(keyer.key(),"Result")

    keyer.set_video(all_green)
    self.assertEqual(keyer.key(),"Result")

    keyer.set_video(most_green)
    self.assertEqual(keyer.key(),"Result")


  def test_detect_foreground_absolute(self):
    pass

  def test_detect_foreground_reflective(self):
    pass

  def test_supress_color_spill(self):
    pass

  def test_create_mask(self):
    pass

if __name__ == '__main__':
    unittest.main()