from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import scipy as scipy

#visualizing things
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from numpy import *
import numpy as np

import json
from pprint import pprint

with open('Desktop/train.json') as data_file_train:    
    trndata = json.load(data_file_train)
with open('Desktop/test.json') as data_file_test:    
    tstdata = json.load(data_file_test)
#
#print len(trndata[0]["ingredients"])
#print len(trndata[90]["ingredients"])
#ingredlength = []
#for n in range(0,len(trndata)):
#    ingredlength.append(len(trndata[n]["ingredients"]))
#print "max", max(ingredlength)
#print "min", min(ingredlength)
#print ingredlength.index(1)
#print trndata[940]
#print ingredlength.index(65)
#print trndata[15289]
#print ingredlength.index(5)


cuisineList = []
ingredientList = []
for n in range(0,len(trndata)):
    cuisineList.append(trndata[n]["cuisine"])
    for m in range(0,len(trndata[n]["ingredients"])):
        ingredientList.append(trndata[n]["ingredients"][m])

# make them unique
from collections import OrderedDict
unique_cuisine = list(OrderedDict.fromkeys(cuisineList))
unique_cuisine_length = len(unique_cuisine)
print 'Unique cuisines: ', unique_cuisine_length  #20

ingred_length = len(ingredientList)
unique_ingreds = list(OrderedDict.fromkeys(ingredientList))
unique_ingreds_length = len(unique_ingreds)
#print ingredList
print 'Unique ingredients: ', unique_ingreds_length  #6714

recipes = [item['ingredients'] for item in trndata]
recipes_length = len(recipes)
print "Number of recipes", recipes_length  #39774
cuisine = [item['cuisine'] for item in trndata]
print 'Number of cuisines', len(cuisine)  #39774


### use this!!
#big_data_matrix = []
#big_data_matrix.append([1,2,3])
#big_data_matrix.append([4,5,6])
#print big_data_matrix
#print big_data_matrix[1]


#zeros = [0] * 100
#print zeros

#sorry, only vague enough for me to remember whats going on :P
#enumerate each ingredient and store in vector
#then for each recipe, put 1 where ingredient exist and 0 if not
#
#big_data_matrix = scipy.sparse.dok_matrix((len(trndata), len(unique_ingreds)), dtype=np.dtype(bool))

big_data_matrix = []
counter = 0
dcounter = 0
for d,dish in enumerate(recipes):  #39774
    #dcounter += 1
    #print 'dcounter', dcounter
    ingred_exists = [0] * unique_ingreds_length
    for i,ingredient in enumerate(unique_ingreds):  #6714
        if ingredient in dish:
            ingred_exists[i] = 1
    big_data_matrix.append(ingred_exists)
    counter += 1
    print "Progress... ", counter, " / ", recipes_length
print 'success!'
##
#
counter = 0
ds = ClassificationDataSet(unique_ingreds_length, 1 , nb_classes=unique_cuisine_length)
for k in xrange(recipes_length): 
    ds.addSample(big_data_matrix[k],cuisine[k])
    counter += 1
    print "added into dataset", counter, " / ", len(cuisine)
print 'classification dataset done!'
#    
#    
#tstdata, trndata = ds.splitWithProportion( 0.25 )
#tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)
#
#tstdata = ClassificationDataSet(len(unique_ingreds), 1 , nb_classes=len(cuisineList))
#for n in xrange(0, tstdata_temp.getLength()):
#    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
#
#trndata = ClassificationDataSet(len(unique_ingreds), 1 , nb_classes=len(cuisineList))
#for n in xrange(0, trndata_temp.getLength()):
#    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
# 
#trndata._convertToOneOfMany( )
#tstdata._convertToOneOfMany( )
#
#print trndata['input'], trndata['target'], tstdata.indim, tstdata.outdim
#
#fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
#trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 
#
##save data
##if  os.path.isfile('oliv.xml'): 
##    fnn = NetworkReader.readFrom('oliv.xml') 
##else:
##    fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
##NetworkWriter.writeToFile(fnn, 'oliv.xml')
#
#trainer.trainEpochs (350)
#print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
#           dataset=tstdata )
#           , tstdata['class'] )