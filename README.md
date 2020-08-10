## Project title
Chroma Keying, an implemention of chroma keying using purely image processing techniques.

## Motivation
Chroma keying is a popular technique used in photography/videography to composite multiple images and video stream together. Chroma keying feature is offered in a lot of video post-processing products such as Adobe After Effects, Premiere Pro, Apple Final Cut Pro X,...However, these products are expensive, therefore, this project could be a free alternative for students doing school works or learning.

## Tech/framework used
- Python
- OpenCV
- Image processing

## Features
- Allow either automatic or manual background detection/removal
- User-friendly manual background detection by using scribbles and trackbars
- Accurate background removal in automatic mode

## How to use?
TODO

## Limitations
- Required the background to be uniform in color, smooth in texture.
- Automatic background detection only works when the background covers the majority of the area in the image.
- Works poorly with partially transparent foreground object since the alpha channel estimation functionality is missing.

## What's next?
- Implement alpha channel estimation for better background removal consistency
- Implement seamless cloning of the foreground into different background image/video
- Take the idea from this project and create a 

## Credits
Big thanks to @spmallick for providing resourceful learning materials. 
This project is an implementation of this research paper: https://pdfs.semanticscholar.org/bf62/cae2be4b4785f4a54ba9d9ef54e0ba61068c.pdf
