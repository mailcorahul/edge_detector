import cv2
import sys , os
import math
import itertools
from operator import itemgetter
import ast
import numpy as np
from os import listdir
from os.path import isfile, join

def detect_lines(img) :

	line_priori = [] ;
	params = [] ;
	h , w = img.shape[:2] ;
	pts = [[[0,0]] , [[0,h]] , [[w,0]] ,[[w,h]]];
	angleDiff =  0.0723599 ; # +- 1.5 degrees threshold
	line_seg = [] ;

	# choose no. of votes based on w & h of the image 
	votes = min(w,h) / 5 ; # use 1/4 th of the image
	lines = cv2.HoughLines(img,1,np.pi/180,votes) ;
	
	if lines is None :
		return pts ;

	i = 0 ;
	for line in lines :
		i = i + 1 ;
		for rho,theta in line:
			add = True ;
			
			# ignoring similar lines
			for r , t in params : 
				if (abs(rho - r) <= 30 and abs(theta - t) <= 0.5) :
					add = False ;
					break ;

				if ( abs(rho - abs(r)) <= 30 or abs(abs(rho) - r) <= 30 ) and abs(theta - t) >= 3 :
					add = False ;
					break ;
				
			if add :
				# including only lines parallel/perpendicular to text orientation

				params.append( [rho,theta] ) ;
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 5000*(-b))
				y1 = int(y0 + 5000*(a))
				x2 = int(x0 - 5000*(-b))
				y2 = int(y0 - 5000*(a))
				cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
				line_seg.append( [x1,y1,x2,y2] ) ;
				line_priori.append(1) ;

	return find_best_quad(line_seg, line_priori , img) ;

# coefficients of x : A , y : B and constant : C 
def lines(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

# using Cramer's Rule to solve linear equations
def intersection(L1, L2, w, h):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        if( x > w or x < 0 or y > h or y < 0 ):
        	return False ;
        return [x,y] ;
    else:
        return False

def findIntersect(line, w, h) :

	pts = [] ;
	for i in range(len(line) - 1) :
		for j in range(i + 1 ,len(line)) :
			inner = [] ;
			x1 , y1 , x2 , y2 = line[i] ; 
			x3 , y3 , x4 , y4 = line[j] ; 
			l1 = lines( [x1,y1], [x2,y2] ) ;
			l2 = lines( [x3,y3], [x4,y4] ) ;
			pt = intersection( l1,l2 ,w ,h) ;
			if pt != False :
				#inner.append(pt) ;
				pts.append(pt) ;

	return pts ;

def lineDist(p1,p2) :

	x = (p2[0] - p1[0]) ** 2 ;
	y = (p2[1] - p1[1]) ** 2 ;
	return math.sqrt(x + y) ;

def find_best_quad(line,line_priori ,img) :

	h , w = img.shape[:2] ;
	quads = [] ;
	isThree = False ;
	extremes = [] ;
	maxArea = -10000 ; max2Area = -10000 ;
	thresholdArea = ( w * h ) / 4 ;
	maxQuad = [] ; max2Quad = [] ;
	bounds = [] ;
	priori = [] ;
	bounds.append([0,0,0,h]) ; # left
	bounds.append([0,0,w,0]) ; # top
	bounds.append([w,0,w,h]) ; # right
	bounds.append([0,h,w,h]) ; # bottom
	horz = False ; vert = False ;
	fullImage = np.array( [[[0,0]] , [[0,h]] , [[w,h]] ,[[w,0]] ] ) ;
	
	# include boundary lines and set vote as 0 
	line.append(bounds[0]) ;
	line.append(bounds[1]) ;
	line.append(bounds[2]) ;
	line.append(bounds[3]) ;

	line_priori.append(0) ;
	line_priori.append(0) ;
	line_priori.append(0) ;
	line_priori.append(0) ;
		
	line_set = range( 0, len(line) ) ;
	for cc in itertools.combinations(line_set , 4):
		list(cc) ;
		i = cc[0] ; j = cc[1] ; k = cc[2] ; l = cc[3] ;
		quad = findIntersect([line[i],line[j],line[k],line[l]],w,h) ;
		priori_sum = line_priori[i] + line_priori[j] + line_priori[k] + line_priori[l] ;
		
		if len(quad) == 4 :
			hull = cv2.convexHull(np.array(quad)) ;
			area = cv2.contourArea(hull) ;
			if len(hull) == 4 : 
				if area >= thresholdArea :
					quads.append(hull) ;
					priori.append(priori_sum) ;
						
	mArea = -1 ; miArea = -1 ;		
	if len(quads) > 0 :				
		maxP = max(priori) ;
		maxiP = priori.index(max(priori)) ;
		for i in range(len(priori)) :
			if priori[i] == maxP :
				tArea = cv2.contourArea(quads[i]) ;
				if tArea > mArea :
					mArea = tArea ;
					miArea = i ;

	return quads[miArea] ;				

def find_boundary(img, edged):
	
	h2 , w2 = img.shape[:2] ;
	h1 , w1 = edged.shape[:2] ;

	quad = np.array(detect_lines(edged)) ;
	cont = img.copy() ;
	cv2.drawContours(cont,[quad], -1, (0,0,255), 3);

	pts = np.float32(quad.reshape((quad.shape[0], quad.shape[2])));	
	pts = sorted(pts, key=itemgetter(1));
	_pts1 = sorted(pts[:2], key=itemgetter(0));
	_pts2 = sorted(pts[2:], key=itemgetter(0));
	pts = np.float32([_pts1[0],_pts2[0], _pts1[1], _pts2[1]]);

	pts = np.float32(pts) ;
	wpts = np.float32([[0,0],[0,h2],[w2,0],[w2,h2]]) ;
	M = cv2.getPerspectiveTransform(pts, wpts) ;
	dst = cv2.warpPerspective(img,M,(w2,h2)) ;
	
	return cont, dst ;

