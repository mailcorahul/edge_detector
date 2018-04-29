import cv2
import numpy as np
import edge_detector
import boundary_detector
import argparse
from PIL import Image

parser = argparse.ArgumentParser() ;
parser.add_argument('--path', type=str, help='input image path') ;
args = parser.parse_args() ;

if __name__ == '__main__' :


	# read image
	img = cv2.imread(args.path) ;

	# detect edges in the image
	edged = edge_detector.detect_edges(img) ;	
	
	# find quadrilateral using houghline transform
	bounded, transformed = boundary_detector.find_boundary(img, edged) ;
	
	# save image
	cv2.imwrite('output/edged.png', edged) ;
	cv2.imwrite('output/bounded.png', bounded) ;
	cv2.imwrite('output/transformed.png', transformed) ;


