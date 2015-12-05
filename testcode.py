from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import scipy as scipy
import time

#visualizing things
#from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
#from scipy import diag, arange, meshgrid, where
#from numpy.random import multivariate_normal
from numpy import *
import numpy as np

import json
from pprint import pprint

##-------------------------
# Control panel
##-------------------------
train_file = 'Desktop/train.json'
test_file = 'Desktop/test.json'

with open(train_file) as data_file_train:    
    trn = json.load(data_file_train)
#with open('M:/Canopy/test.json') as data_file_test:    
#    tstdata = json.load(data_file_test)

# 39774 recipes
# about 40,000 recipes is quite a lot...
# lets make it smaller ~ get first 1000

number_of_recipes_to_use = 500
trn = trn[0:number_of_recipes_to_use]
print 'new dataset size: ', len(trn)

cuisineList = []
ingredientList = []
for n in range(0,len(trn)):
    cuisineList.append(trn[n]["cuisine"])
    for m in range(0,len(trn[n]["ingredients"])):
        ingredientList.append(trn[n]["ingredients"][m])

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

recipes = [item['ingredients'] for item in trn] 
recipes_length = len(recipes)
print "Number of recipes", recipes_length  #39774
cuisine = [item['cuisine'] for item in trn]
print 'Number of cuisines', len(cuisine)  #39774

#enumerate each ingredient
#then for each recipe, put 1 where ingredient exist and 0 if not
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
    print "Recipes progress... ", counter, " / ", recipes_length
print 'recipes success!'
print 'time taken: ', time.clock()   # in seconds, or milliseconds if fast

#cuisine matrix
cuisine_matrix = []
counter = 0
for c,cuis in enumerate(cuisine):
    cuisine_type = [0]*unique_cuisine_length
    for t,types in enumerate(unique_cuisine):
        if types in cuis:
            cuisine_type[t] = 1
    cuisine_matrix.append(cuisine_type)
    counter += 1
    print "Cuisines progress... ", counter, " / ", recipes_length
print 'cuisines success!'

print 'time taken: ', time.clock()  # in seconds
# can probably merge the upper 2?

counter = 0
ds = ClassificationDataSet(unique_ingreds_length, unique_cuisine_length , nb_classes=unique_cuisine_length)
for k in xrange(recipes_length): 
    ds.addSample(big_data_matrix[k],cuisine_matrix[k])
    counter += 1
    print "added into dataset... ", counter, " / ", recipes_length
# No memory error when using a desktop with 16g ram
print 'classification dataset done!'  
print 'time taken: ', time.clock()
    

    
tstdata, trndata = ds.splitWithProportion( 0.25 )
tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)

tstdata = ClassificationDataSet(unique_ingreds_length, unique_cuisine_length , nb_classes=unique_cuisine_length)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(unique_ingreds_length, unique_cuisine_length , nb_classes=unique_cuisine_length)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
 
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print 'split = ok'

#print trndata['input'], trndata['target'], tstdata.indim, tstdata.outdim

#save data - replace fnn with this
#if  os.path.isfile('food.xml'): 
#fnn = NetworkReader.readFrom('food.xml') 
#else:
    #fnn = buildNetwork( trndata.indim, 100 , trndata.outdim, outclass=SoftmaxLayer )

fnn = buildNetwork( trndata.indim, 100 , trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 

print 'neural net made.'
print 'training start.'



trainer.trainEpochs (25)  # the more recipes there are, the longer each epoch will take
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
           dataset=tstdata )
           , tstdata['class'] )

NetworkWriter.writeToFile(fnn, 'Desktop/food.xml')
print 'time taken: ', time.clock()

# TRAINING RESULTS
#
# dataset size      epochs trained      percent error(%)
# 500               25                  0.8
# 1000              50                  0.4     ... it might be a 40% error rate...
# 1000              100                 0.4
# 1000              150                 0.4    ..still 0.4% lol. though 99.6% accuracy rate sure is pretty high
# 2000              50                  0.2     pretty sure there are more decimals but 
# 2000              100                 0.2     seems like percentError() truncated it
# 3000              50                  0.13333333333     ...nvm
# 4000              50                  0.1
# 
# using all 39774 recipes will probably take days to train
# 1 epoch took about 15 mins to run

####------------------------------------------------------------------------------------------------

with open(test_file) as data_file_test:    
    tst = json.load(data_file_test)
    
test_recipes = [item['ingredients'] for item in tst] 
test_recipes_length = len(test_recipes)
print "Number of recipes", test_recipes_length  #39774
#cuisine = [item['cuisine'] for item in trndata]
#print 'Number of cuisines', len(cuisine)  #39774
test_recipes = test_recipes[0:2]
    
testing_matrix = []
counter = 0
dcounter = 0
for d,dish in enumerate(test_recipes):  #39774
    #dcounter += 1
    #print 'dcounter', dcounter
    ingred_exists = [0] * unique_ingreds_length
    for i,ingredient in enumerate(unique_ingreds):  #6714
        if ingredient in dish:
            ingred_exists[i] = 1
    testing_matrix.append(ingred_exists)
    counter += 1
    print "Recipes progress... ", counter, " / ", test_recipes_length
print 'recipes success!'
print 'time taken: ', time.clock()   # in seconds, or milliseconds if fast


print '-------------------------------------------------'
print 'all cuisines: ', unique_cuisine

for r in xrange(500):
    #print '1 recipe in testing: ', trn[r]['ingredients']
    cuisine_vector = fnn.activate(big_data_matrix[r])
    max_prob = max(cuisine_vector)
    max_prob_index = cuisine_vector.argmax(max_prob)
    this_cuisine_type = unique_cuisine[max_prob_index]
    #print 'The cuisine type of this recipes is: ', this_cuisine_type 
    #print 'The cuisine should be: ', trn[r]['cuisine']
    #print '-------------------------------------------------'
    print '#, computed, original: ', r, this_cuisine_type, trn[r]['cuisine']