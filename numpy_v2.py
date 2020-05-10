
from __future__ import print_function
import os
import numpy as np

#***********Containers********************
# Containers are data structures holding elements
# and typically hold all their values in memory,e.g lists,dict,tuple
# Technically, an object is a container when it can be asked whether it contains a certain element
# most containers are also iterable

# *************** Iterables *********************
# An iterable is any object, not necessarily a data structure,
# that can return an iterator (with the purpose of returning all of its elements)
# Iterable  generates an Iterator when using  iter() method.
#If an object is iterable, it can be passed to the built-in Python function iter()
# Iterator  iterate over an iterable object using __next__() method.

## list of cities
cities = [1,2,3]
# initialize iterator object
iterator_obj = iter(cities) # type(iterator_obj): list_iterator
elem1 = next(iterator_obj)
elem2 = next(iterator_obj)
elem3 = next(iterator_obj)

# If you want to grab all the values from an iterator at once, you can use the built-in list()/tuple function. Among other possible uses,
# list() takes an iterator as its argument, and returns a list consisting of all the values that the iterator yielded:
# for example, build infinite iterator for odd numbers
iterator_obj1 = iter(cities)
all_elem = list(iterator_obj1)
# Part of the elegance of iterators is that they are “lazy.” That means that when you create an iterator,
# it doesn’t generate all the items it can yield just then.
# It waits until you ask for them with next(). Items are not created until they are requested.

# create class that has _iter_ and _next_ methods
class OddIter:
    def __iter__(self):
        # initial elemnt to start with
        self.num = 1
        return self

    def __next__(self):
        num = self.num
        self.num += 2
        return num

#  create iterator now
a1 = iter(OddIter())
e1 = next(a1)
e2 = next(a1)
# Be careful to include a terminating condition, when iterating over these type of infinite iterators.

#************* Generators*******************
#A generator allows you to create iterators , in an elegant succinct syntax
# that avoids writing classes with __iter__() and __next__() methods.


# generator function
# uses the Python yield keyword instead of return
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
# yield indicates where a value is sent back to the caller, but unlike return, you don’t exit the function afterward.
# Instead, the state of the function is remembered. That way, when next() is called on a generator object
# (either explicitly or implicitly within a for loop),
# the previously yielded variable num is incremented, and then yielded again

# usage:
gen = infinite_sequence()
e1 = next(gen)
e2 = next(gen)
e3 = next(gen)

#********* generator expression:allow you to quickly create a generator object in just a few lines of code. **********
# advantage: you can create them without building and holding the entire object in memory before iteration
# ****# for example:
# write a expression to square set of numbers
# using list
nums_squared_lc = [num**2 for num in range(5)]
# using generator by using ()
nums_squared_gc = (num**2 for num in range(5))
nums_squared_lc1 = list(nums_squared_lc)

# ******************different forms of for loop************************

# When a for loop is executed, for statement calls iter() on the object, which it is supposed to loop over.
# If this call is successful, the iter call will return an iterator object that defines the method __next__(),
# which accesses elements of the object one at a time

#for <var> in <iterable>:
   # <statement(s)>



# python range :The range type represents an immutable sequence of numbers( is a type like list and tuple)

rng1 = list(range(5)) ## Prints out 3,4,5
rng2 = list(range(3, 8, 2)) # range(st,stop,delta): 3,5,7


itm = np.arange(4)
for i in itm:
    print(i)

# uses array as iterable seq
itm1 = [8 ,9,10]
for i in itm1:
    print(i)
# here, for i in [list]/array is used, but when using range, list(range)
# is not nedded

 #  uses range as iterable seq
for i in range(3, 8, 2):
    print(i)

# using enumerate:Accessing each item in a list (or another iterable),Also getting the index of each item accessed
start_indx =0 # will aceess each elemnt, starting counter at this index number
for indx,i in enumerate(itm1,start_indx): #
    print(indx,i) # prints 0,8..1,9...2,10, but with start_indx=2, gives 2,8..3,9..4,10

# numpy  np.ndenumerate(arr):  iterator yielding pairs of array coordinates and values for ND array
# can be used with 1 d array like enumerate too.
nd1 = np.array([[1, 2], [3, 4]])
for index, x in np.ndenumerate(nd1):
    print(index, x)
#(0, 0) 1
#(0, 1) 2
#(1, 0) 3
#(1, 1) 4

#************* python zip: similar to enumerate but is most useful for acessing multiple lists *************
# returns an iterator of tuples based on the iterable objects.
# If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.
# If multiple iterables are passed, zip() returns an iterator of tuples with each tuple having elements from all the iterables.
group = ['A','B','C']
tag = ['a','b','c']
for idx, x in enumerate(group):
    print(x, tag[idx])
# prints
# A a
# B b
# C c
# same can be accompolished by zip

for x, y in zip(group, tag):
    print(x, y)
# Note: Izip is similar to zip, but zip combines all elements at one time(more memory), but izip does that one by one

#

#****************** creating array inside loop ****
# python append and extend
# append: Appends object at the end of LIST.( length of list only increases by 1)
# extend:Extends list by appending elements from the iterable (( length of list increseas by num of elements in iterable)
x = [1, 2, 3]
x.append([4, 5])  # [1,2,3,[4,5]]
x1 = [1, 2, 3]
x1.extend([4, 5])
x2 = [1, 2, 3]
x2 =  x2 + [4,5]  # same as extend

#****** however, numpy append will append to an array/list: np.append(arr or even list,elements to append,axis=..)

#
# create array [0,1,2,3,4] inside loop
x_arr_ins_loop = np.array([])
#x= np.array([0,1,2,3])
for i in range(4):
    x_arr_ins_loop = np.append(x_arr_ins_loop, i)

# *********dictionary   ***********
# empty dict
dic_empty =  dict()
# created with {key1:val1,key2:val2..}
dic1 = {1:'a',2:'b',3:'c'}
# key with multiple values
dic2 = {1:['a','b'],2:['c','d']}

# items() gives class dict_items , which you can't acess individually, as is but
# can access within for loop
items1 = dic1.items()
print ("Dict key-value are : ")
for k1, v1 in dic1.items():
    print (k1, v1)
# gives 1 a
      # 2 b
      # 3 c
#** ***create list of dict keys and values from dictionary
def read_dict(dic):
    k2_arr = np.array([])
    v2_arr = np.array([])
    for k2, v2 in dic.items():
        k2_arr = np.append(k2_arr,k2)
        v2_arr = np.append(v2_arr, v2)
    return k2_arr,v2_arr
k2_arr,v2_arr = read_dict(dic2)

 #**** create dictionary from keys(list) and multiple values(tuple) , using loop
k3 = [1,2]
v3 = ['a','b','c','d']
def creat_dict(k,v):
    lbl_dict =  dict()
    cnt = 0
    for i in k:
        lbl_dict.update({i: tuple([v[cnt],v[cnt+ 1]])})
        cnt = cnt + 2
        #lbl_dict.update({k[i]: tuple(v[i])})
    return lbl_dict
dic4 = creat_dict(k3,v3)



# *******check if current path exists,else create directory: os.path.exists(),os.makedirs()*********
pth = 'C://ML//env//tf//test_numpy'
if not os.path.exists(pth):
    os.makedirs(pth)
#********** get most nested folder from path : os.path.basename(), path upto last folder:os.path.dirname()   *****************
pth_folder_nm = os.path.basename(pth)
pth_bs = os.path.dirname(pth)
pth_bs_oneup = os.path.dirname(os.path.dirname(pth))
# better method 2: use pathlib library:  use Path().parents[i]
from pathlib import Path
pth_bs_v2 = Path(pth).parents[0]
pth_bs_oneup_v2 = Path(pth).parents[1]

# join 2 paths **** os.path.join(pth1,pth2)*****************

# ****create list of all files matching extension in  in a folder: use glob.glob(folder_pth + "//*.file_ext") ***
import glob
pth1 = os.getcwd()
py_files = glob.glob(pth1 + "//*.py")


# list all subdir in current dir: os.listdir()
dirs = os.listdir(pth)

# ***********os.walk(pth)  **** gives generator , with 3 tuples back
#   ##

for pth_upto_subdir, subdir_nm, files in os.walk(pth, topdown=False):
    print(pth_upto_subdir)
    print(subdir_nm)
    print(files)
# C://ML//env//tf//test_numpy\f1
# []
# ['areds_v1.py', 'speed_challenge_v1.py']
# C://ML//env//tf//test_numpy\f2
# []
# ['areds_v2.py', 'speed_challenge_v2.py']
# C://ML//env//tf//test_numpy\f3
# []
# ['areds_v3.py', 'speed_challenge_v3.py']
# C://ML//env//tf//test_numpy
# ['f1', 'f2', 'f3']
# []

 # ****** create list of all files matching extension in a folder and its subdirectories ****
path_list_files = []
for path2, subdirs1_categ, files1 in os.walk(pth):
    if len(files1) != 0:
        path_list_files = path_list_files + files1

print('****************')
# bubbsort()
bb1= np.random.randint(10,20,10)
def swap(x,y):
    temp = x
    y = x
    x= temp
    return x,y
def bbsort(x):
    for i in range(np.size(x)-1,0,-1):
        sorted = 1
        for j in range(i):
            sorted = 0
            if (x[j] > x[j+1]):
                 x[j],x[j+1] = swap(x[j],x[j+1])
                 sorted = 0

bbsort(bb1)


#********** slection sort ******
predicted = np.array([4, 25,  0.75, 11])
observed  = np.array([3, 21, -1.25, 13])
diff = predicted - observed
r1 = np.std(diff)
r2 =  r1*r1

from sklearn.metrics import mean_squared_error
from math import sqrt

rms1 = sqrt(mean_squared_error(observed, predicted))
print('h')



#