import matplotlib.pyplot as plt
import numpy as np
import pdb
import random
import copy

animate = True
#######################################################################################
# input parameters
#######################################################################################
# meshgrid for checkerboard
width = 100
height = 100
x = np.linspace(0,1, width)
y = np.linspace(0,1, height)
X,Y = np.meshgrid(x,y)
img = np.zeros((width, height))
N_pix = width*height

# volume fraction for two phases, phase 2 is black pixels
phi_1 = 0.5 		# volume fraction of whites
phi_2 = 1 - phi_1	# volume fraction of blacks

# theoretical S_2(r)
def S2_true(rr_range, phi_1, phi_2):
	Debye = False
	if Debye:
		a = 2.0  # correlation length scale, in number of pixels
		return phi_1*phi_2*np.exp(- rr_range /a ) + phi_2**2   # for Debye random media: cf. (Yeong & Torquato 1998)
	else:
		a = 8
		return phi_1*phi_2*np.exp(- rr_range /a )*np.cos(rr_range) + phi_2**2 

#######################################################################################
# actual calculations
#######################################################################################

num_black_pix = int(phi_2*N_pix)
# randomly choose UNIQUE indices of black pixels in a 1D array
black_indices = random.sample(range(N_pix), num_black_pix)
white_indices = []
for index in range(N_pix):
	if index not in black_indices:
		white_indices.append(index)
#######################################################################################
## step 1: initialize checkerboard
# convert each indices into number of row and column, starting from bottom-left corner
# then construct an initial random checkerboar with given volume fraction
def row_col(index):
	col = index%width
	row = int((index - col)/width)
	return row, col

for index in black_indices:
	col = index%width
	row = int((index - col)/width)
	img[row,col] += 1

plt.figure(figsize=(7,7))
plt.pcolor(X,Y, img, cmap='Greys')
plt.show()
#######################################################################################
# step 2: initialize 2-point probability function
sampling_range = 20		      # sampling range
r_range = np.arange(sampling_range)       # sampling lengths in terms of number of pixels, same in both directions
S2_rows = np.zeros((height, sampling_range))
S2_cols = np.zeros((width, sampling_range))
for r in r_range:
	# first sweep rows
	for j in range(height):   	
		cl = 0
		while cl<width: # stop at the right boundary, go to next row
			neigh = cl+r
			if neigh>=width:   # impose periodic boundary
				neigh -= width
			if img[j,cl]*img[j,neigh]>0:
				S2_rows[j,r] += 1.0
			cl += 1           # go 1 pixel to the right, reiterate
		
	# second sweep columns
	for i in range(width):
		rw = 0
		while rw<height: # stop at the top boundary, go to next column 
			neigh = rw + r
			if neigh>=height:
				neigh -= height
			if img[rw,i]*img[neigh,i]>0:
				S2_cols[i,r] += 1.0
			rw += 1           # go 1 pixel up, reiterate

# now average two S2 contributions, normalize by number of pixels
S2 = (np.sum(S2_rows,axis=0)/float(N_pix) + np.sum(S2_cols, axis=0)/float(N_pix))/2.0

plt.figure(figsize=(7,7))
plt.plot(r_range, S2, 'k')
plt.scatter(r_range, S2, color='r')
plt.xlabel(r'$\rm r$', fontsize=25)
plt.ylabel(r'$\rm S_2(r)$', fontsize=25)
plt.xlim([np.min(r_range), np.max(r_range)])
plt.ylim([0,1])
plt.tight_layout()
plt.show()

#######################################################################################
### iterate 3 and 4 until convergence of S_2(r) at current stage to theoretical S_2(r)
# 3. Interchange random pixel pairs: one from each phase, use Metropolis algorithm to accept or discard
# 4. re-evaluate S_2(r) by only looking at row and column of interchanged pairs
#######################################################################################
def update_S2(Image, S2_Row, S2_Col, black_ind, white_ind): 
	# evaluate new S2_row and S2_col after interchange
	b_row, b_col = row_col(black_ind)
	w_row, w_col = row_col(white_ind)
	image = copy.deepcopy(Image)
	S2_row = copy.deepcopy(S2_Row)
	S2_col = copy.deepcopy(S2_Col)
	# interchange black with white
	image[b_row, b_col] = 0.0
	image[w_row, w_col] = 1.0
	# set two rows and columns to 0 for S2_row, S2_col	
	S2_row[w_row,:] = 0.0
	S2_row[b_row,:] = 0.0
	S2_col[w_col,:] = 0.0
	S2_col[b_col,:] = 0.0
	# re-evaluate these rows and columns
	for r in r_range:
		# first sweep rows
		for j in list(set([w_row, b_row])):
			cl = 0
			while cl<width: # stop at the right boundary, go to next row
				neigh = cl + r
				if neigh>=width:
					neigh -= width
				if image[j,cl]*image[j,neigh]>0:
					S2_row[j,r] += 1.0
				cl += 1           # go 1 pixel to the right, reiterate
		# second sweep columns
		for i in list(set([w_col, b_col])):
			rw = 0
			while rw<height: # stop at the top boundary, go to next column 
				neigh = rw + r
				if neigh>=height:
					neigh -= height
				if image[rw,i]*image[neigh,i]>0:
					S2_col[i,r] += 1.0
				rw += 1           # go 1 pixel up, reiterate
	# compute new S2
	S2_new = (np.sum(S2_row,axis=0)/float(N_pix) + np.sum(S2_col, axis=0)/float(N_pix))/2.0
	return S2_row, S2_col, S2_new, image
#######################################################################################
k = 0   # number of interchanged pairs
num_shuffle = 0
unsuccessful_interchange = 0
energy = 10
tol = 1e-3
S2_target = S2_true(r_range, phi_1, phi_2)
random.shuffle(white_indices)
if animate:
	plt.ion()
	fig = plt.figure(figsize=(7,7))
	im = plt.pcolor(X,Y,img, cmap='Greys')
	plt.draw()
while energy>tol and unsuccessful_interchange<20000:
	# shuffle pair orders when pairs are repeating
	
	# blacks and whites are now different indices!
	if k>=int(np.min((phi_1,phi_2))*N_pix):
		random.shuffle(black_indices)
		random.shuffle(white_indices)
		k = 0
		num_shuffle += 1
	
	# 3. Interchange random pixel pairs: one from each phase, use Metropolis algorithm to accept or discard
	# first choose a pair:
	black = black_indices[k]
	white = white_indices[k]
	# evaluate new S_2(r) after interchanging the pair values
	new_S2_rows, new_S2_cols, new_S2, new_image = update_S2(img, S2_rows, S2_cols, black, white)
	#pdb.set_trace()	
	# evaluate \Delta E 
	E = np.sum((S2 - S2_target)**2)
	Ep = np.sum((new_S2 - S2_target)**2)
	dE = Ep - E
	# compute Temp such that initial acceptance rate = 0.5 = exp(-dE/Temp)
	if k==0:
		Temp = 0.2*abs(dE/np.log(0.5))
	else:
		Temp *= 0.95    # T(k)=T(0)*lambda**k annealing schedule. (Jiao Yang 2019)
	# apply Metropolis method
	if dE <=0:
		Prob = 1.0
	else:
		Prob = np.exp(-dE/Temp)
	rnd = np.random.rand()
	if rnd <= Prob:
		# accept interchange black with white
		S2_rows = copy.deepcopy(new_S2_rows)
		S2_cols = copy.deepcopy(new_S2_cols)
		S2 = copy.deepcopy(new_S2)
		black_indices[k] = white
		white_indices[k] = black
		w_row, w_col = row_col(white)
		img[w_row, w_col] = 1.0
		b_row, b_col = row_col(black)
		img[b_row, b_col] = 0.0
		energy = Ep
	else:
		unsuccessful_interchange += 1		
	k += 1
	if animate:
		im.set_array(img.ravel())
		#im.set_data(img)
		fig.canvas.draw()
		fig.canvas.flush_events()
	print('Shuffles: ', num_shuffle, ' iteration: ', k, ' dE: ', dE, ' energy: ', energy , ' unsuccessful interchange: ', unsuccessful_interchange)

if not animate:
	plt.figure(figsize=(7,7))
	plt.pcolor(X,Y,img, cmap='Greys')
	plt.show()

plt.figure(figsize=(7,7))
plt.plot(r_range, S2, 'k', label=r'$\rm prediction$')
plt.plot(r_range, S2_target, color='r', label=r'$\rm target$')
plt.xlabel(r'$\rm r$', fontsize=25)
plt.ylabel(r'$\rm S_2(r)$', fontsize=25)
plt.xlim([np.min(r_range), np.max(r_range)])
plt.ylim([0,1])
plt.legend()
plt.tight_layout()
plt.show()
#########################################################################

pdb.set_trace()
