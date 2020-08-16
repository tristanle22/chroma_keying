import cv2
import numpy as np
import matplotlib.pyplot as plt 
import json
from sketcher import Sketcher

from mpl_toolkits.mplot3d import Axes3D
  
class ChromaKeyer:
  """
  A chroma keying class. This chroma keying method is implemented based on global color statistic. The following
  assumptions of the background is critical for this method to work:
    1. Textureless
    2. A uniform color
    3. Is the most prominent color
    4. Foreground does not contain background color

  For reference, visit this paper:
  https://pdfs.semanticscholar.org/bf62/cae2be4b4785f4a54ba9d9ef54e0ba61068c.pdf

  Attributes
  ----------
  video: cv2.VideoCapture()
    The video stream that contains the foreground and background region
  auto_background_detection_enable: bool
    The flag which indicates whether to use auto/manual background extraction
  background_hue_range: tuple
    The (lower,upper) hue range of background color
  variance_amp: float
    The ampilifying factor of variance calculated in histogram analysis method
  threshold_variance_delta: float 
    The threshold for magnitude of change in variance calculated in histogram analysis method
  threshold_refine_background: int
    The upper range threshold used for refining the detected background region
  threshold_foreground_absolute: int
    The threshold for which a pixel is considered absolute foreground
  threshold_foreground_reflective: int
    The threshold for which a pixel is considered reflective foreground
  saturation_lightness_function: np.ndarray
    The mapping of lightness (V-channel) to saturation (S-channel)

  Methods
  -------
  update_settings(config_file)
    Update the attributes with the settings from the configuration file
  
  key()
    Main method, stream and extract background from video source
  
  alpha_channel_estimation() - TODO
    Estimate value of the alpha channel - determine which pixel should be transparent (background), which pixel should
    be opague (foreground)

  manual_background_detection()
    Use interactive user interface to extract background
  
  auto_background_detection(frame)
    Automatically detect background region in a frame

  get_background_mask_in_range(frame)
    Acquire the background mask based on the hue range

  detect_foreground(frame)
    Detect different region of the foreground

  detect_foreground_absolute(frame)
    Detect absolute foreground region

  detect_foreground_reflective(frame)
    Detect reflection foreground region - i.e. foreground regions that reflect the color of background

  color_spill_suppression(frame)
    Suppression color spill from background to foreground

  histogram_analysis(frame, channel, mask)
    Run histogram analysis on input to extract the predominant colors

  get_hue_range(histogram)
    Extract the continuous color range that corresponds to the most dominant color from the histogram

  apply_mask(image, mask)
    Helper function to apply mask onto image

  build_GMM_color_model()
    Build a Gaussian Mixture Model that represents the colors of the image with a few dominant colors

  lightness_grouping(frame)
    Group pixels of the frame into different channels, which correspond to a range of lightness value
  """

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

    if settings_path:
      self.update_settings(settings_path)

  def update_settings(self, settings_path):
    """@brief Update settings of ChromaKeyer object from a json file.
              Helps to keep the settings tidy

      @param settings_path Path to the settings json file
    """
    if not settings_path:
      return False

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

    return True
    
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
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

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
    return self.refine_background(frame, background_mask)

  def refine_background(self, frame, mask):
    """@brief Refine the background mask, for dealing with transparent regions

       @param frame The video frame being refined. Used for calculating gradient
       @param mask The background mask to be modified
       @return The refined background mask
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    gradient = cv2.Sobel(frame[:,:,2],cv2.CV_32F,1,1)
    gradient = np.abs(gradient)

    fine_background_mask = cv2.inRange(gradient,0,self.threshold_refine_background)
    mask_refined = cv2.bitwise_and(mask, fine_background_mask)

    return mask_refined

  def get_background_mask_in_range(self, frame):
    """@brief Obtain background mask based on pre-determined hue range

       @return Background mask
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    frame_background_mask = cv2.inRange(frame[:,:,0],
                                        self.background_hue_range[0],
                                        self.background_hue_range[1])
    return self.refine_background(frame,cv2.bitwise_not(frame_background_mask))

  def detect_foreground(self, frame):
    """@brief Detect different foreground region of the image

       @param foreground Foreground subtracted from background
       @return The mask for absolute foreground region,
               The mask for reflective foreground region AND
               The foreground with green intensity suppressed in BGR
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    absolute_foreground_mask = self.detect_foreground_absolute(frame)
    reflective_foreground_mask = self.detect_foreground_reflective(frame)

    green_suppressed_foreground_bgr = self.color_spill_suppression(cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))

    return (absolute_foreground_mask,reflective_foreground_mask,green_suppressed_foreground_bgr)

  def detect_foreground_absolute(self, frame):
    """@brief Identify absolute background that is far enough 
              from the background hue range

       @param frame The video frame being used
       @return The foreground mask of absolute foreground region
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

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

  def detect_foreground_reflective(self, frame):
    """@brief Detect colorless region based on saturation threshold
     
       @param foreground The foreground portion of the photo in BGR
       @return The mask of reflective foreground region
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    saturation_threshold_matrix = cv2.LUT(frame[:,:,2],self.saturation_lightness_function)
    gray_confidence = saturation_threshold_matrix - frame[:,:,1]

    reflective_foreground_mask = cv2.inRange(gray_confidence,self.threshold_foreground_reflective,255)

    return reflective_foreground_mask

  def color_spill_suppression(self, frame):
    """@brief Suppress green color spilling in the colorless region
     
       @param frame The colorless portion of the photo in BGR
       @return The frame with green spilling suppressed in BGR
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      print("Invalid input")
      return np.zeros_like(frame)

    b,g,r = cv2.split(frame)
    max_br = np.maximum(b,r)
    g[g > max_br] = max_br[g > max_br]

    return cv2.merge((b,g,r))

  def histogram_analysis(self, frame, channel, mask):
    """@brief Extract dominant pixels in a channel of a frame

       @param frame The video frame in HSV being used for histogram analysis
       @param channel The channel chosen for histogram analysis (H,S,V)
       @param mask The mask to extract region of interest (ROI) from the frame
       @return An numpy array that contains only the most dominant bins
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3 or frame.shape[:2] != mask.shape or channel not in range(3):
      return (0,0)

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
    
    return self.get_hue_range(background_hue)

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
    if image.shape[:2] != mask.shape
      print("Invalid input, image shape != mask shape")
      return False

    return cv2.bitwise_and(image,image,mask=mask)

  def build_GMM_color_model(self):
    ret, frame = self.video.read()
    
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = frame_hsv.astype(np.float32)
    frame_hsv /= 255.0

    lightness_groups = self.lightness_grouping(frame_hsv)
    for x in lightness_groups:
      hist = cv2.calcHist([frame_hsv], [0,1], x, [256,256], [0.0, 1.0, 0.0, 1.0])
      cv2.imshow("hist", hist)
      cv2.waitKey(0)

  def lightness_grouping(self, frame):
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return []

    lightness_masks = []
    step = 0.1
    lower_lim = 0.0
    upper_lim = 1.0
    for i,j in zip(np.arange(lower_lim, upper_lim, step), np.arange(lower_lim+step, upper_lim+step, step)):
      mask = cv2.inRange(frame[:,:,2], i, j)
      lightness_masks.append(mask)
    return lightness_masks
