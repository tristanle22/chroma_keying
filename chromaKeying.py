import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 
import json
from sketcher import Sketcher
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

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
    """
    Parameters
    ----------
    video: cv2.VideoCapture
      The video capture that contains foreground and monochromatic (usually green) background
    auto_back_det_enable: bool
      The flag that enables auto (True) or manual (False) background detection
    settings_path: str
      The path to the configuration file
    """

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
    """ Update settings of ChromaKeyer object from a json file. Helps to keep the settings tidy

    Parameters
    ----------
    settings_path: str
      The path to the settings file
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
    """Perform chroma keying on the video footage.
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
    """Detect bacground leveraging user input. Start by drawing scribbles onto the background in
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
    """Automatically detect background of an image based on global color statistic, 
       assumptions:
        1. Background is textureless
        2. Background is uniform in color
        3. Background is the most prominent color
        4. Foreground does not contain background color

      Parameters
      ----------
      frame: np.ndarray

      Returns
      -------
      bool ndarray
        The mask for refined background
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
    """ Refine the background mask, for dealing with transparent regions

    Parameters
    ----------
    frame: np.ndarray 
      The video frame being refined. Used for calculating gradient
    mask: bool np.ndarray 
      The background mask that indicate which region to be modified
    
    Returns
    -------
      Refined background mask
    """
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    gradient = cv2.Sobel(frame[:,:,2],cv2.CV_32F,1,1)
    gradient = np.abs(gradient)

    fine_background_mask = cv2.inRange(gradient,0,self.threshold_refine_background)
    mask_refined = cv2.bitwise_and(mask, fine_background_mask)

    return mask_refined

  def get_background_mask_in_range(self, frame):
    """Obtain background mask based on pre-determined hue range

    Parameters
    ----------
    frame: np.ndarray
      The frame where background color range is extracted from
    
    Returns
    -------
    bool np.ndarray
      The refined background mask
    """
    
    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    frame_background_mask = cv2.inRange(frame[:,:,0],
                                        self.background_hue_range[0],
                                        self.background_hue_range[1])
    return self.refine_background(frame,cv2.bitwise_not(frame_background_mask))

  def detect_foreground(self, frame):
    """Detect different foreground region of the image

    Parameters
    ----------
    frame: np.ndarray
      The frame which contain the foreground object
    
    Returns
    -------
    bool np.ndarray
      The mask for absolute foreground region
    bool np.ndarray
      The mask for reflective foreground region
    np.ndarray
      The foreground with green-intensity suppressed, in BGR format
    """

    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    absolute_foreground_mask = self.detect_foreground_absolute(frame)
    reflective_foreground_mask = self.detect_foreground_reflective(frame)

    green_suppressed_foreground_bgr = self.color_spill_suppression(cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))

    return (absolute_foreground_mask,reflective_foreground_mask,green_suppressed_foreground_bgr)

  def detect_foreground_absolute(self, frame):
    """Identify absolute background, that is the color region far enough 
       from the background hue range

    Parameters
    ----------
    frame: np.ndarray 
      The video frame being used
    
    Returns
    ------- 
    bool np.ndarray
      The foreground mask of absolute foreground region
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
    """Detect colorless region based on saturation threshold

    Parameters
    ----------
    frame: np.ndarray 
      The foreground portion of the photo in BGR

    Returns
    -------  
      The mask of reflective foreground region
    """

    if len(frame.shape) < 3 or frame.shape[2] < 3:
      return False

    saturation_threshold_matrix = cv2.LUT(frame[:,:,2],self.saturation_lightness_function)
    gray_confidence = saturation_threshold_matrix - frame[:,:,1]

    reflective_foreground_mask = cv2.inRange(gray_confidence,self.threshold_foreground_reflective,255)

    return reflective_foreground_mask

  def color_spill_suppression(self, frame):
    """Suppress green color spilling in the foreground colorless region
    
    Parameters
    ----------
      frame: np.ndarray 
        The colorless portion of the photo in BGR
    Returns
    -------
    np.ndarray  
      The frame in which green spilling has been suppressed, BGR format
    """

    if len(frame.shape) < 3 or frame.shape[2] < 3:
      print("Invalid input")
      return np.zeros_like(frame)

    b,g,r = cv2.split(frame)
    max_br = np.maximum(b,r)
    g[g > max_br] = max_br[g > max_br]

    return cv2.merge((b,g,r))

  def histogram_analysis(self, frame, channel, mask):
    """Extract dominant colors in a channel of a frame

    Parameters
    ----------
    frame: np.ndarray 
      The video frame in HSV being used for histogram analysis
    channel: int 
      The channel chosen for histogram analysis (0:H, 1:S, 2:V)
    mask: bool ndarray 
      The mask that identifies ROI to run histogram analysis on

    Returns
    -------
    tuple
      A tuple that contains the lower and upper bound of the dominant color range
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

  def get_hue_range(self, histogram): # TODO Change naming of this function
    """Extract the range of bins that contains the tallest bin

    Parameters
    ----------
    histogram: np.ndarray 
      The histogram that contains all the bins
    Returns 
    -------
    tuple 
      A tuple of min and max indices of the 
    """

    i = j = np.argmax(histogram)
    while (histogram[i] > 0 or histogram[j] > 0) and (i > 0 and j < histogram.shape[0]-1):
      if i > 0 and histogram[i] > 0:
        i -= 1
      if j < histogram.shape[0]-1 and histogram[j] > 0:
        j += 1

    return (i.item(),j.item())

  def apply_mask(self, image, mask):
    """ Utility function used for applying mask on an image

    Parameters
    ----------
    image: np.ndarray
      The image
    mask: bool np.ndarray
      The mask

    Returns
    -------
    np.ndarray
      The image with mask applied, a.k.a Region of Interest (ROI)
    """

    if image.shape[:2] != mask.shape:
      print("Invalid input, image shape != mask shape")
      return False

    return cv2.bitwise_and(image,image,mask=mask)

  def build_GMM_color_model(self):
    ret, frame = self.video.read()
    
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = frame_hsv.astype(np.float32)
    frame_hsv /= 255.0

    lightness_groups = self.lightness_grouping(frame_hsv)
    k_size = 5
    for i,x in enumerate(lightness_groups):
      print(f"Lightness group: {i}")
      hist = cv2.calcHist([frame_hsv], [0,1], x, [256,256], [0.0, 1.0, 0.0, 1.0])    

      hist_Median = cv2.medianBlur(hist, 5) # TODO: replace median blur with morph or thresholding?
      smoothed_hist = self.smoothen_histogram(hist_Median)

      local_max, labels = self.extract_color_distribution(smoothed_hist, 30)
      row, col = np.nonzero(local_max)
      # print("Peak pixel index: {}".format((row,col)))
      # print("Peak pixel values: {}".format(smoothed_hist[row, col]))
      # print("-------------------------------")
      peaks = self.apply_mask(smoothed_hist, local_max.astype(np.uint8))
 
      output = np.concatenate([hist, smoothed_hist, peaks], axis=1)
      cv2.imshow("Original - Smoothed - Peaks", output)
      if cv2.waitKey(0) & 0xFF == 27:
        break

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

  def smoothen_histogram(self, histogram, gamma=10):
    n, d = histogram.shape
    if n == 0 or d == 0 or n != d:
      print("Histogram must be a non-empty, square matrix")
      return np.array([])
    
    I = np.identity(n)
    D1 = np.diff(I,axis=0)
    D2 = np.diff(I, 2, axis=0)

    P = gamma**2 * np.dot(D2.T, D2) + 2 * gamma * np.dot(D1.T, D1)
    return np.linalg.solve((I + P), histogram)

  def extract_color_distribution(self, hist, min_distance=20):
    local_max_raw = peak_local_max(hist, min_distance=min_distance,
                                     threshold_abs=2, 
                                     exclude_border=5, 
                                     indices=False)

    peaks = self.NMS_peaks(self.apply_mask(hist, local_max_raw.astype(np.uint8)),min_distance)
    local_max_refined = np.zeros_like(local_max_raw)
    local_max_refined[peaks] = 1

    markers, num_markers = ndimage.label(local_max_refined, structure=np.ones((3,3)))
    labels = watershed(-hist, markers)

    print("Number of peaks: {}".format(np.count_nonzero(local_max_refined)))
    # print(np.unique(markers))
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    return (local_max_refined, labels)
  
  def NMS_peaks(self, peaks, threshold):
    if np.count_nonzero(peaks) == 0:
      return ([],[])

    row, col = np.nonzero(peaks)
    new_x, new_y = [row[0]], [col[0]]

    for c in zip(row, col):
      unique = True
      for x,y in zip(new_x, new_y):
        if np.sqrt((c[0]-x)**2 + (c[1]-y)**2) <= threshold:
          unique = False
          if peaks[c] > peaks[x,y]:
            x = c[0]
            y = c[1]
            break
      if unique:
        new_x.append(c[0])
        new_y.append(c[1])

    return (new_x,new_y)

def main():
  filename = './data/toronto.jpg'
  filename = os.path.abspath(os.path.join(os.path.dirname(__file__),filename))

  cap = cv2.VideoCapture(filename)
  settings_file = os.path.abspath(os.path.join(os.path.dirname(__file__),"chromaKeySettings.json"))
  keyer = ChromaKeyer(cap, True, settings_file) # Change to 

  # keyer.key()
  keyer.build_GMM_color_model()
  # keyer.test_2d_hist()
  # keyer.smoothen_histogram_penalized_likelihood(None, 2)

if __name__ == '__main__':
  main()
