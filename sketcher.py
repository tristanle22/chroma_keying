import cv2

"""@brief Sketcher class, a user interface for background color selection
"""
class Sketcher:
  def __init__(self, windowname, dests):
    self.prev_pt = None
    self.windowname = windowname
    self.dests = dests
    self.show()
    cv2.setMouseCallback(self.windowname, self.on_mouse)

  def show(self):
    cv2.imshow(self.windowname, self.dests[0])
    # cv2.imshow(self.windowname+"_mask",self.dests[1])

  def on_mouse(self, event, x, y, flags, param):
    pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
      self.prev_pt = pt
    elif event == cv2.EVENT_LBUTTONUP:
      self.prev_pt = None

    if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
      for dst in self.dests:
          cv2.line(dst, self.prev_pt, pt, (255,255,255), 5)
      self.prev_pt = pt
      self.show()
