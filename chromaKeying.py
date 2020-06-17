import cv2
import numpy as np
import matplotlib.pyplot as plt 
import json
from sketcher import Sketcher

"""@brief A chroma keying class. This chroma keying method is based on global color statistic and has the following
          assumption on the background:
            1. Textureless
            2. A uniform color
            3. Is the most prominent color
            4. Foreground does not contain background color
          
          The implementation follows this research paper:
          https://pdfs.semanticscholar.org/bf62/cae2be4b4785f4a54ba9d9ef54e0ba61068c.pdf
"""
class ChromaKeyer:
  def __init__(self, video, auto_back_det_enable, settings_path):
    self.video = video
    self.auto_background_detection_enable = auto_back_det_enable
    
    self.background_hue_range = (0,0)

    self.variance_amp = 0
    self.threshold_variance_delta = 0
    self.threshold_refine_background = 0
    self.threshold_foreground_absolute = 0
    self.threshold_foreground_reflective = 0
    self.saturation_lightness_function = None

    self.update_settings(settings_path)

  def update_settings(self, settings_path):
    """@brief Update settings of ChromaKeyer object from a json file.
              Helps to keep the settings tidy

      @param settings_path Path to the settings json file
    """
    with open(settings_path,'r') as f:
      settings = json.load(f)
      
      self.variance_amp = settings['variance_amp']
      self.threshold_variance_delta = settings['threshold_variance_delta']
      self.threshold_refine_background = settings['threshold_refine_background']
      self.threshold_foreground_absolute = settings['threshold_foreground_absolute']
      self.threshold_foreground_reflective = settings['threshold_foreground_reflective']
      self.saturation_lightness_function = np.interp(np.arange(0,256),
                                                     settings['saturation_lightness_function']['x_pivots'],
                                                     settings['saturation_lightness_function']['y_pivots'])

  def key(self):
    """@brief Perform chroma keying on the video footage.
    """
    if self.auto_background_detection_enable:
      while(self.video.isOpened()):
        ret, frame = self.video.read()
        if ret:
          frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
          background_mask = self.auto_background_detection(frame)
          foreground = self.apply_mask(frame, background_mask)
          cv2.imshow("Chroma Keyed", cv2.cvtColor(foreground,cv2.COLOR_HSV2BGR))
          k = cv2.waitKey(25) &0xFF
          if k == 27:
            break
          elif k == ord('p'):
            cv2.waitKey(0)
        else:
          print("can't read frame")
          break
    else:
      self.manual_background_detection()
      while (self.video.isOpened()):
        ret, frame = self.video.read()
        if ret:
          background_mask = self.get_background_mask_in_range(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV))
          foreground = self.apply_mask(frame, background_mask)
          cv2.imshow("Chroma Keyed", foreground)
          k = cv2.waitKey(25) &0xFF
          if k == 27:
            break
          elif k == ord('p'):
            cv2.waitKey(0)
        else:
          print("can't read frame")
          break

    cv2.destroyAllWindows()

  def alpha_channel_estimation(self):
    #TODO Implement alpha channel estimation
    pass

  def manual_background_detection(self): #TODO: Implement status return AND user instruction
    """@brief Detect bacground leveraging user input. Start by drawing scribbles onto the background in
              the Color_select frame. Background hue range will be calculated by the pixels masked by the scribbles
    """
    def do_nothing(*args):
      pass

    settings_window = "Settings"
    color_select_window = "Color_select"
    tolerance_slider_name = "Tolerance"
    foreground_window_name = "Foreground"

    ret,frame = self.video.read()

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.namedWindow(settings_window)
    cv2.namedWindow(color_select_window)
    cv2.createTrackbar(tolerance_slider_name,settings_window,0,255,do_nothing)
    cv2.imshow(foreground_window_name,np.zeros_like(frame_hsv[:,:,0]))

    frame_color_select = frame.copy()
    frame_color_mask = np.zeros(frame.shape[:2],dtype='uint8')
    frame_background_mask = np.zeros(frame.shape[:2],dtype='uint8')
    sketcher = Sketcher(color_select_window,[frame_color_select,frame_color_mask])

    while True:
      k = cv2.waitKey(25) & 0xFF
      if k == 27:
        break
      else:
        background_color = self.apply_mask(frame_hsv[:,:,0], frame_color_mask)
        background_color_list = background_color[background_color > 0]
        if background_color_list.any():
          mean_background_color = np.mean(background_color_list)
          tolerance = cv2.getTrackbarPos(tolerance_slider_name,settings_window)

          self.background_hue_range = (np.clip(mean_background_color-tolerance,0,255),
                                      np.clip(mean_background_color+tolerance,0,255))
          frame_background_mask = self.get_background_mask_in_range(frame_hsv)
          cv2.imshow(foreground_window_name, self.apply_mask(frame, frame_background_mask))

    cv2.destroyAllWindows()
  
  def get_background_mask_in_range(self, frame):
    """@brief Obtain background mask based on pre-determined hue range

       @return Background mask
    """
    frame_background_mask = cv2.inRange(frame[:,:,0],
                                        self.background_hue_range[0],
                                        self.background_hue_range[1])
    return self.refine_background(frame,cv2.bitwise_not(frame_background_mask))

  def auto_background_detection(self, frame):
    """@brief Automatically detect background of an image based on
              global color statistic, assumming the background:
              1. Textureless
              2. A uniform color
              3. Is the most prominent color
              4. Foreground does not contain background color

        @param frame The video frame that is used for background detection
        @return result The background mask that blocks out background and
                       keep foreground
    """
    h_channel,s_channel,v_channel = cv2.split(frame)
  
    # Hue channel
    h_min, h_max = self.histogram_analysis(frame,0,None)
    self.background_hue_range = (h_min,h_max)
    mask_h = cv2.inRange(h_channel, h_min, h_max)

    # Saturation channel
    s_roi = cv2.bitwise_and(s_channel,s_channel,mask=mask_h)
    s_min, s_max = self.histogram_analysis(frame,1,mask_h)
    mask_s = cv2.inRange(s_roi, s_min, s_max)

    # Lightness channel
    v_roi = cv2.bitwise_and(v_channel,v_channel,mask=mask_h)
    v_min, v_max = self.histogram_analysis(frame,2,mask_h)
    mask_v = cv2.inRange(v_roi,v_min,v_max)

    # Refine background using gradient threshold
    background_mask = cv2.bitwise_not(mask_h*mask_s*mask_v)
    result = self.refine_background(frame, background_mask)

    return result

  def detect_foreground(self, foreground):
    """@brief Detect different foreground region of the image

       @param foreground Foreground subtracted from background
       @return The mask for absolute foreground region,
               The mask for reflective foreground region AND
               The foreground with green intensity suppressed in BGR
    """
    absolute_foreground_mask = self.detect_foreground_absolute(foreground)
    reflective_foreground_mask = self.detect_foreground_reflective(foreground)

    green_suppressed_foreground_bgr = self.color_spill_suppression(cv2.cvtColor(foreground, cv2.COLOR_HSV2BGR))

    return (absolute_foreground_mask,reflective_foreground_mask,green_suppressed_foreground_bgr)

  def refine_background(self, frame, mask):
    """@brief Refine the background mask, for dealing with transparent regions

       @param frame The video frame being refined. Used for calculating gradient
       @param mask The background mask to be modified
       @return The refined background mask
    """
    gradient = cv2.Sobel(frame[:,:,2],cv2.CV_32F,1,1)
    gradient = np.abs(gradient)

    fine_background_mask = cv2.inRange(gradient,0,self.threshold_refine_background)
    mask_refined = cv2.bitwise_and(mask, fine_background_mask)

    return mask_refined

  def detect_foreground_absolute(self, frame):
    """@brief Identify absolute background that is far enough 
              from the background hue range

       @param frame The video frame being used
       @return The foreground mask of absolute foreground region
    """
    h_min, h_max = self.background_hue_range
    foreground_hue = []
    
    if h_min < self.threshold_foreground_absolute and h_max < 255 - self.threshold_foreground_absolute:
      foreground_hue = [(0,h_min - self.threshold_foreground_absolute),(h_max + self.threshold_foreground_absolute, 255)]
    elif h_max > 255 - self.threshold_foreground_absolute:
      foreground_hue = [(h_max + self.threshold_foreground_absolute - 255, h_min - self.threshold_foreground_absolute)]
    else:
      foreground_hue = [(h_max + self.threshold_foreground_absolute, 255 + h_min - self.threshold_foreground_absolute)]

    foreground_mask = np.zeros(frame.shape[:2], dtype='uint8')
    for i in foreground_hue:
      foreground_mask = cv2.bitwise_or(foreground_mask,cv2.inRange(frame[:,:,2],i[0],i[1]))

    return foreground_mask

  def detect_foreground_reflective(self, foreground):
    """@brief Detect colorless region based on saturation threshold
     
       @param foreground The foreground portion of the photo in BGR
       @return The mask of reflective foreground region
    """
    saturation_threshold_matrix = cv2.LUT(foreground[:,:,2],self.saturation_lightness_function)
    gray_confidence = saturation_threshold_matrix - foreground[:,:,1]

    reflective_foreground_mask = cv2.inRange(gray_confidence,self.threshold_foreground_reflective,255)

    return reflective_foreground_mask

  def color_spill_suppression(self, frame):
    """@brief Suppress green color spilling in the colorless region
     
       @param frame The colorless portion of the photo in BGR
       @return The frame with green spilling suppressed in BGR
    """
    b,g,r = cv2.split(frame)
    max_br = np.maximum(b,r)
    g[g > max_br] = max_br[g > max_br]

    return cv2.merge((b,g,r))

  def histogram_analysis(self,frame,channel,mask):
    """@brief Extract dominant pixels in a channel of a frame

       @param frame The video frame in HSV being used for histogram analysis
       @param channel The channel chosen for histogram analysis (H,S,V)
       @param mask The mask to extract region of interest (ROI) from the frame
       @return An numpy array that contains only the most dominant bins
    """
    hist = cv2.calcHist([frame], [channel], mask, [256], [0,256])
    background_hue = np.zeros_like(hist,dtype='float')

    var_0 = np.var(hist)
    var_k = var_0
    var_prev = var_k

    for i in range(256):
      max_indices = np.argmax(hist)
      background_hue[max_indices] = hist[max_indices]
      hist[max_indices] = 0
      var_k = np.var(hist)
      if var_k == 0 or var_k < self.variance_amp * var_0 and abs(var_k-var_prev)/var_prev < self.threshold_variance_delta:
        break
      var_prev = var_k
    results = self.get_hue_range(background_hue)
    return results

  def get_hue_range(self, histogram):
    """@brief Extract the group consecutive bins which contains the tallest bin

       @param histogram The histogram that contains all the bins
       @return A tuple of min and max bin indices
    """
    i = j = np.argmax(histogram)
    while (histogram[i] > 0 or histogram[j] > 0) and (i > 0 and j < histogram.shape[0]-1):
      if i > 0 and histogram[i] > 0:
        i -= 1
      if j < histogram.shape[0]-1 and histogram[j] > 0:
        j += 1
    return (i.item(),j.item())

  def apply_mask(self, image, mask):
    return cv2.bitwise_and(image,image,mask=mask)
