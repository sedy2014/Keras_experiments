from __future__ import print_function
import numpy as np

print("hello{name1")
#***********print the full numpy array without truncating ******************
np.set_printoptions(threshold=6)
#************** create array **********
# method1 : from list
x_lst = [0,1,2,3,4,5,6,7,8,9]
x_arr = np.array(x_lst)
x_arr1= np.array(x_lst,dtype='float',copy=True)
x_arr2 = x_arr.astype(('float'))  # change array type
x_arr3 = x_arr.astype(('float')).astype(('int64'))
x_arr_to_list = x_arr2.tolist()
# filling numpy array from strings
x_arr4 = np.zeros((3,1),dtype = object)
x_arr4 = ['a','b','c']
#*********** Creating NumPy array from  mix of List and Tuples ******
a_list = [1, 2.5, 3]
a_tuple = (1.5 , 2.3, 3)
two_d_list_tuple_array = np.array([a_list, a_tuple])

# ***********Create a 1D array of numbers from 0 to 9: ***************
x1 = np.arange(10) # x1 = 0:9
print("x1.shape: " + str(x1.shape)) # (10,)
print("x1.ndim: " + str(x1.ndim)) #  1
print("x1.size: " + str(x1.size)) #10
print("x1.data: " + str(x1.data)) # <memory at 0x0000021FF15AA708
print("x1.dtype: " + str(x1.dtype)) # int32
print("x1.strides: " + str(x1.strides)) # (4,)

# np.arange(st,end,del) = st,st+del ... end-1 == [st :end-1:del] matlab , #ele= roundtoinfinity(end-st/del)
# for count down, last element is end+1 , but numv=ber of elements is end+st/del
x2 = np.arange(1,10,1)  # x2 = 1:9
x3 = np.arange(1,10,2)  # x2 = 1:2:9
x3_1 = np.arange(1,11,2)  # x2 = 1:2:9
x4 = np.arange(10,-1,-1)  # last element = -1+1 = 0, and #n = 10+1/1 = 11
x5 = np.arange(10,-2,-1)  # last element = -2 + 1 = -1, and #n = 10+2/1 = 12
x6 = np.arange(10,-3,-2)  # last element = -3 + 1 = -2, and #n = 10+3/2 = [6.5] = 7

#*** using linspace ********************
x3_1_1 = np.linspace(start= 1,stop = 10,num = 5,endpoint=False) # del = end-st/N = 9/5=1.8 so, 1:1+1.8=2.8,
x3_1_2 = np.linspace(start= 1,stop = 10,num = 5) # del = end-st/N-1 == 9/4 =2.25 so 1,1+2.25...

#**************** slicing *********************
#   x1[m:n] == m to n-1 ==matlab a(m+1:n)
#   x1[:n] = 0 to n-1 == matlab a(1:n)
#    x1[n:] =  n to last element   == matlab a(n+1:end)
print(x1[2:5])  # x1(2,3,4) == [[2 3 4]]
print(x1[-1])       # last element: x1(end) 9
print(x1[-2])     #   x1(end-1) : 8
print(x1[-2:])       # last 2 elements: x1(end-1:end), in general: x1([-n:1])  == last n lements == matlan:x1(end-n+1:end)
print(x1[3:])       # a[3:last]  or matlab a(n+1:end) : [3 4 5 6 7 8 9]
print(x1[:0])    #[]== x1([0:-1])
print(x1[:3])      # x1([0:3]) == [0 1 2]
print(x1[::-1])   #[9 8 7 6 5 4 3 2 1 0]== reverse(x1) in matlab

# ************copy array without modifying source
x1_same = np.copy(x1)#  if x1_same = x1, modification of x1_same effect x1 too
x1_same[:]= 3 # set all array values to same  == matlab a(:) = 3

x1_arr = np.array([0,1,2,3,4,5,6,7,8,9])  # sp np.array([]) == np.arange(..) or np.

# *************2d arrays**************
x6 = np.zeros(5) # shape = (5)
# specfy shape as list
x7 = np.zeros([5,1]) # shape = (5,1)
# shape as tuple
x8 = np.zeros((5,1)) # shape = (5,1)

# *** extract element from array using np.where()*********
x1_a= np.arange(4,12,1)
x9 = x1_a[x1_a>5] # [6 7 8 9 10 11]: same as matlab x1((x1>5))
x9_1 = x1_a[np.where(x1_a>5)] # same as x1[x1>5]
x10 = x1_a.take(np.where(x1_a>5))  # [[6 7 8 9 10 11]], so x1[0][0] is 6

# non zero elemnts in array
xz= np.array([0,5,7,0,9])
ind= np.nonzero(xz)  # returns Indices of elements that are non-zero.
xnz = xz[ind]

#*********** stacking  *****************
x10 = np.arange(0,3,1)
x11 = np.arange(3,6,1)
x12 = np.arange(6,9,1)
x13 = np.arange(9,12,1)

# combine row wise
#  np.vstack(tuple of arrays)  == matlab [x1;x2]
xrow1 = np.vstack((x10,x11))
xrow2 = np.vstack((x12,x13))
# stack all 4 rows :mthd1
xrow3 = np.vstack((xrow1,xrow2))
# # stack all 4 rows :mthd2
xrow4 = np.vstack(((x10,x11),(x12,x13)))

# combine column wise,( concatanate)
xcol1 = np.hstack((np.hstack((x10,x11)),np.hstack((x12,x13))))  # [ 0,1,2,3,4,5,6,7,8,9,10,11]
xcol2 = np.hstack(((x10,x11),(x12,x13)))  # combine two rows, then combine results  as columns
# combine column  wise, from rows
xcol3 = np.vstack(((x10,x11),(x12,x13))).T # matlab [x1;x2]'

# create 2-D array of N rows within loop
def create_row_arr_within_loop(N):
    result_array = np.empty((0,N))
    for line in range(3):
        result = np.random.choice(3,3)
        result_array = np.vstack((result_array, result))


# *********create 2-d arrays :=Method 2
#  create each element as list
# so np.array([a,b]) is 1d array
#  now np.array([[...],[...]] is row wise array, each row = list
x2d_1 = np.array([[1 ,2, 3],[4 ,5, 6],[7,8,9],[10,11,12]])
# if needed to get data column wise, write it correctly
# e.g [ 1 4;  2 5; 3 6]
x2d_2 = np.array([[1 ,4],[2 ,5],[3,6]])
# method 3: np.arange(#N).reshape(m,n) where m*n = N
print("x2d_1.shape: " + str(x2d_1.shape))
print("x2d_1.ndim: " + str(x2d_1.ndim))
print("x2d_1.size: " + str(x2d_1.size))
print("x2d_1.data: " + str(x2d_1.data))
print("x2d_1.dtype: " + str(x2d_1.dtype))
#The stride value represents the number of bytes that must be travelled in memory in order to reach
# #the next value of an axis of an array.
# for int32( 4 bytes),for axis 0 or row, we can reach next element [0,0] to [1,0] after traversing
# column elemnts = 3 of them , each of 4 bytes== 12, and along axis 1, move from one column to another
# bt moving 4 bytes , so stride=(12,4)

x2d1_strd = x2d_1.strides





# for 3-d array, axis 0 means along row, but skipping 2d array, axis 1 is rowise, axis 2 is climn wise
x3d = np.array([[[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]],
       [[ 4,  5,  6,  7],
        [12, 13, 14, 15]]])
x3d1_strd = x3d.strides




#**************slicing 2d array *****************
# [m,n] = elemnt at m-1 row, n-1 col
x2d_s1 = x2d_1[1,2]
# row 0 == matlab (1,:)
x2d_s2 = x2d_1[0,] # is treated as [0,:]
# colmun 0 == matlab (:,1)
x2d_s3 = x2d_1[:,0]  # cannot use [,0], needs explicit :
# all except first m rows == matlab (m+1:end,:)
x2d_s4 = x2d_1[2:,]
# last two row  == matlab (end-1:end,:)
x2d_s5 = x2d_1[-2:,]
# all except first m columns == matlab (:,m+1:end)
x2d_s6 = x2d_1[:,2:]
# last two columns
x2d_s7 = x2d_1[:,-2:]
# every other column [:,::m] is every mth column
x2d_s8 = x2d_1[:,::2]
# row r1:r2, column c1 to c2 [r1:r2+1,c1:c2+1]
x2d_s9 = x2d_1[0:2,1:3]

# mean of entire 2-d array
m1= np.mean(x2d_1)
# mean  column wise(axis 0 wise)
m2 = np.mean(x2d_1,axis=0)
# mean of row wise
m3 = np.mean(x2d_1,axis=1)
# mean/min/max
# rowwise




#**************** random integers***************
#np.random.rand(m,n): random numbers(float) between (0,1] of shape m,n
#np.random.randint(a,b,(m,n)): random integers between (a,b) of shape m,n
#np.random.randint(a,b,m): random integers between (a,b) of size m(1-d)
# same as np.random.choice(a, b)

r1= np.random.rand(2,2)
r2= np.random.randint(2,10,(2,2))
# np.random.randn(a,b) # random floats from N(0,1) of shape m,n
# mu + sigma*np.random.randn(a,b) # random floats from N(mu,sigma) of shape m,n
r3 = np.random.randn(10000,1)
mu_r3= np.mean(r3)
sigma_r3= np.std(r3)
r4 = 2+ 0.5*np.random.randn(10000,1)
mu_r4= np.mean(r4)
sigma_r4= np.std(r4)


#np.random.choice(a,m) # choose m samples between 0 and a, where samples can be duplicate(with replacement)
# np.random.choice(a,m,replace=False) # choose m samples between 0 and a, without replacement
n1 = np.random.choice(5,3)
n2 = np.random.choice(5,3,replace = False)

#*********** shuffle an array of numbers ***************
x_un = np.arange(10,21)
# method1
N = np.prod(x_un.shape)  # number of elemnts in array== np.size()
x_shff1 = np.random.choice(x_un,size=N,replace=False)
# method 2( create Random indices of same size as array)
rand_ind = np.random.choice(N,N,replace = False)
x_shff2 = np.take(x_un,rand_ind)

# *************transpose and reshape ****************
# reshape
r1_tr = np.transpose(r1)
# by default, axis 0 is row, axis 1 is column, when transpoing
# if you want row and colum swapped, you change axis 1 and 0 so, (0,1)->(1,0)
# strides for axis are changed too
r1_strd = r1.strides
r1_tr_strd = r1_tr.strides
r1_tr1 = np.transpose(r1,axes=(1,0))  # same as r1_tr

# reshape 2d array
r2= np.arange(12).reshape(3,4)
# in numpy array os shape (N) or (N,) is 1-Dim, treated as flat array
# while (N,1) is column vector of 2Dim ( 2,1) and (1,N) is row vector of 2 Dim
#to column
r2_rshp = r2.reshape(r2.size,1)
r2_rshp_sh = r2_rshp.shape
r2_rshp_dim = r2_rshp.ndim
# flattened
r2_rshp1 = r2.reshape(r2.size,)
r2_rshp_sh1 = r2_rshp1.shape
r2_rshp_dim1 = r2_rshp1.ndim
r2_flttn1 = r2.flatten()
r2_flttn2 = r2.flatten(order='C')
r2_flttn2_sh1 = r2_flttn2.shape
r2_flttn2_dim2 = r2_flttn2.ndim

# reshape 2d array to row
r2_rshp2 = r2.reshape(1,r2.size)

# flatten picking up from original array columwise
r2_rshp3 =  np.transpose(r2).reshape(r2.size)
r2_flttn3 = r2.flatten(order='F')
# ravel is same as flatten, but flatten returns copy, so original is not modified if flattened is modified,
# but with ravel, changing flattened array modifies original too: so ravel is inplace flatten

# **********reshape 1d array to 2d, *********
r3 = np.array([1,2,3,4,5,6])
r3_2d = r3.reshape(2,3)

# extract odd elemnts from array
f1= np.array([1,2,3,4,5,6])
f2= f1[f1%2==1]

# repeating sequence
r4 = np.array([[1,2],[3,4]])
# repeat r1,r2 2 times along axis 0
r4_rp1 = np.repeat(r4, 2,axis=0)
# repeat c1,c2 2 times along axis 1
r4_rp2 = np.repeat(r4, 2,axis=1)
# repeat r1 1 time,r2 2 times, along axis 0
r4_rp3 = np.repeat(r4, [1,2],axis=0)
# tile( copy entire array certain times):numpy.tile(A, reps)
r4_rp4 = np.tile(r4, (2, 2))

a = np.array([1,2,3])
b = np.hstack((np.repeat(a,3),np.tile(a,3)))

# common and not common items between numpy aray
c= np.array([11, 2, 13 ,4 ,15, 6])
d= np.array([11, 8, 13 ,10 ,15, 11])

# numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
# Return the sorted, unique values that are in both of the input arrays.
comm = np.intersect1d(c,d)
comm1= c[np.where(c==d)]

# not matching :numpy.setdiff1d(ar1, ar2, assume_unique=False)Return the unique values in ar1 that are not in ar2.
not_comm1= np.setdiff1d(c, d)
not_comm2= c[np.where(c!=d)]

# create array, whose each elemnt is a tuple, so treated as structure
# following create array of 2 values, each tuple with 3 elements, in,float and string
# each of struture elemen can be named with a field as ('field','datatype')
x_arr_of_tup = np.array([(1,2.,'Hello'), (5,3.,"World")], dtype=[('elem1','int32'),('elem2','float'), ('elem3','S10')])
x_arr_of_tup0= x_arr_of_tup[0]
# acess each element of array tuple based on its field /name
x_arr_of_tup_elem1 = x_arr_of_tup['elem1']
x_arr_of_tup_elem2  = x_arr_of_tup['elem2']
x_arr_of_tup_elem3 = x_arr_of_tup['elem3']

# method2
x_arr_of_tup_elem3_v2 = np.array([row[2] for row in x_arr_of_tup])

# for example get species names from following data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
species = iris_1d['f4']
species1 = np.array([row[4] for row in iris_1d])

#********* np.argmax:Returns the indices of the maximum values along an axis.*************
#In  multiple occurrences , the indices corresponding to the first  are returned.
# for no axis, dor N dim array, index after flatenning returned
bb = [1, 5, 5, 5, 4, 5, 5, 2, 2, 2,5]
argmx = np.argmax(bb) # max is 5, at index 1(first occurence)
bb_2d = np.array([[10, 11, 12], [13, 14, 15]])
argmx1 = np.argmax(bb_2d)  # max value 15 at index 5(after flattening)
# checking index without flatenning
argmx2 = np.unravel_index(np.argmax(bb_2d),bb_2d.shape)

# ************ find the count of unique values in a numpy array *************
bb = [1, 5, 5, 5, 4, 5, 5, 2, 2, 2,5]
b_uniq,unq_indc,un_cnt = np.unique(bb, return_index=True,return_counts=True)
# b_uniq[0]= 1 whcih has first occurence at index 0(unq_inc[0]), and count 1(un_cnt[0])
# b_uniq[3]= 5 whcih has first occurence at index 1(unq_inc[3]), and count 6(un_cnt[3])

#************"Most frequent value in the above array ****************
# find index of max of uniq count array, and acess original array at that index
b_mst_freq = b_uniq[np.argmax(un_cnt)]

# **********np.bincount(x):: gives how many times each index value occurs in a array *****
# e.g [ 1 3 1 5]: index 0: 0 not there, so bincount =0: index 1: 2 times, index 2 : 2 not there: bincount 0

# *******From the array a, replace all values greater than 5 to 5 and less than 2 to 2. *****
# method1
t1= np.random.randint(0,7,20)
t2 = np.copy(t1) # for refrence
t_ref= np.copy(t1)
t3= np.vstack((t1>5,t1<2))
t1[t3[0,:]]=5
t1[t3[1,:]]=2
# method2 : using np.clip
# b = numpy.clip(a, a_min, a_max,) # clip the array to a_min and a_max
# note: np.clip(a,a_min,a_max,a) will do inplace clip
np.clip(t2,2,5,t2)


#*********sort array : np.sort(arr,axis = ) ***************
# e.g [9,4,15,0,7] can be sorted as [ 0,4,7,9,15] so, its indices
# from original array are: 4,1,
r5 =  np.array([9,4,15,0,7])
r5_sorted = np.sort(r5)
# r5_sorted = r5[r5_sorted_indx]
r5_sorted_indx = np.argsort(r5)

# ******** 2d sort  **************
r6 = np.array([[ 9,  4, 15,  0, 17],[16,17,8,9,0]])
# default, sorts each row
r6_sort = np.sort(r6)
# sort column
r6_sort_col = np.sort(r6,axis=0)
#sort entire 2d array and shape it as before
r6_sort_all =  np.sort(r6.flatten()).reshape(r6.shape)

# ******** euclidean distance between two arrays: compute norm of difference ********
#np.linalg.norm(x, ord=, axis=): computes norm of x, over axis(if specfied)
# ord = None specified( L2) else specify 'fro'== L2,1(L1 norm)
d1 =  np.array([1,2,3,4,5])
d2 =  np.array([4,5,6,7,8])
eucl_dist = np.linalg.norm(d1-d2)

# ******** moving average of a numpy array ************
mv = np.array([0,1,2,3,4,5,6,7,8])
N= 3 # length pf window for moving average
# mthd1
mv_avg1 = np.convolve(mv, np.ones((N,))/N, mode='full')


# *************matrix/array manipulations  *****************
# multiply arrays elemntwise
# np.multiply(x1, x2,) == x1*x2
y1= np.array([0,1,2])
y2=  np.array([1,2,3])
y3 = np.multiply(y1,y2)
y3_1 = y1*y2

# multiply matrices elemntwise
y4 = np.array([[1,2],[3,4]])
y5= np.array([[5,6],[7,8]])
y6 = np.multiply(y4,y5)
y7 = y4 * y5

# multiply matrices normally  USe: np.matmul(, ) == x1@x2
y8 = np.matmul(y4,y5)
y8_1 = y4 @ y5

# determinant of matrix :np.linalg.det
det = np.linalg.det(y4)
# eignevalue and eignevectors of matrix :np.linalg.eig
ev,egvc = np.linalg.eig(y4)

# inverse of matrix :np.linalg.inv
y4_inv = np.linalg.inv(y4)
print('h')
