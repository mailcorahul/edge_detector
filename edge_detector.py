import cv2
import imutils
import numpy as np
from cv2 import ximgproc
import os

cwd = os.getcwd(); 
path = os.path.join(cwd,'lib/model.yml.gz'); 
pDollar = ximgproc.createStructuredEdgeDetection( path ); # loading the model file for detecting edges. 

def detect_edges(img) :

	h, w = img.shape[:2] ;

	img = np.float32(img); 
	img = img / 255; 
	edges = pDollar.detectEdges(img) ; 
	
	minV = 0 ;	
	maxV = np.max(edges) ;  
	thresh_val = 0.84 ; # edges usually has a intensity greater than 0.84

	# apply average thresholding , if the max thresh in the image is > thresh_val
	if maxV > thresh_val :
		avg = ( maxV - minV ) / 2 ; 
	else :
		avg = thresh_val ;

	edges[edges > avg] = 255; 
	edges[edges <= avg] = 0; 				

	return np.uint8(edges) ;
