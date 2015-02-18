from moi import latpert as lp # modify this or make MOI a module
import pandas as pd
'''
tmts = {"control":range(0,51),"mass":range(52,97),"inertia":range(97,131)}


animIDs = [8, 4, 4, 6, 6, 8, 8, 8, 4, 4, 4, 7, 7, 7, 7, 7, 3, 2, 2, 2, 2, 6, 2,
       2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 9,
       9, 9, 9, 9, 9, 9, 8, 4, 4, 4, 4, 4, 6, 6, 8, 8, 8, 8, 7, 4, 3, 3, 5,
       5, 5, 5, 5, 5, 5, 9, 9, 5, 2, 2, 2, 2, 2, 2, 2, 7, 9, 9, 9, 9, 9, 2,
       8, 8, 4, 6, 6, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 7, 7, 2, 2, 2, 2, 2, 2,
       2, 1, 1, 1, 1, 1, 1, 1, 1, 9, 6, 4, 4, 8, 9, 9]

treatments = pd.DataFrame(columns=('AnimalID','Treatment','Directory'),index=range(0,131))

for trt in tmts:
    for i in tmts[trt]:
        #print i
        treatments.loc[i] = [animIDs[i], trt, lp.rr[i].name]

#print len(animIDs)

treatments.to_pickle('treatments.pckl')
'''

treatments = pd.read_pickle('treatments.pckl')

def findDataIDs(animalID = None, treatmentType = None):
    if animalID == None:
        return list(treatments.query('Treatment == "' + treatmentType + '"').index)
    elif treatmentType == None:
        return list(treatments.query('AnimalID == ' + str(animalID)).index)
    else:
        return list(treatments.query('Treatment == "' + treatmentType + '" and AnimalID == ' + str(animalID)).index)

print treatments

#print treatments.query('Treatment == "control" and AnimalID == 8').index
