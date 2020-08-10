from chromaKeying import ChromaKeyer
import os
import cv2
import numpy as np
import argparse

def main():
  filename = './data/rabbit.mp4'
  filename = os.path.abspath(os.path.join(os.path.dirname(__file__),filename))

  cap = cv2.VideoCapture(filename)
  settings_file = os.path.abspath(os.path.join(os.path.dirname(__file__),"chromaKeySettings.json"))
  keyer = ChromaKeyer(cap, True, settings_file) # Change to 

  # keyer.key()
  keyer.build_GMM_color_model()
  # keyer.test_2d_hist()

if __name__ == '__main__':
  main()
