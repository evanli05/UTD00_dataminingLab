# coding:UTF-8
import numpy as np

from sklearn.cluster import DBSCAN
from numpy import *
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import scipy as sp
import scipy.io as sio

def readTargetData(path):
    feature = []
    label = []
    count = 0
    with open(path) as file:
        for rows in file:
            if count<400:
                process = []
                line = rows.strip().split(',')
                process = [float(x) for x in line]
                #process = [float(x) for x in line[:-1]]
                #print process[-1]
                if process[-1] == 4.0:
                #if line[-1] == 'vacuum_cleaning':
                    count+=1
                feature.append(process[:-1])
                label.append(process[-1])
    return feature,label




def clus_Info(labels,n_clusters,sourceLabel):
    numSamples = len(labels)
    clusterAssment = mat(zeros((numSamples, 1)))

    clusId2label = {}
    for i in xrange(numSamples):
        clusterAssment[i, :] = labels[i]
    #print "clusterAssment",clusterAssment
    for j in range(n_clusters):
        indexInClust = nonzero(clusterAssment[:, 0].A == j)[0]
        total = len(indexInClust) * 1.0
        maxPoint = 0.0
        labelSet = {}
        assignLabel = 0

        for item in indexInClust:
            pointLabel = sourceLabel[item][0]
            if pointLabel not in labelSet.keys():
                labelSet[pointLabel] = 0

            labelSet[pointLabel]+=1

        for key in labelSet:
            if labelSet[key] > maxPoint:
                maxPoint = labelSet[key]
                assignLabel = key

        purity = maxPoint/total
        print "In cluster id = "+str(j)+", assignLabel = "+str(assignLabel)+" purity = ",purity

        clusId2label[j] = assignLabel
        print"-----------------------------------------------------"

    return clusId2label

# Compute DBSCAN
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# #core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# print labels
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print "num of clusters",n_clusters_
if __name__ == '__main__':
    dataName = "Syndata_c5"
    rate = 0.5
    d = 40
    noCluster = 5

    sfPath = 'C:/Users/yifan/OneDrive - The University of Texas at Dallas/Workspace/UTD00_dataminingLab/Code/Data/' + dataName + '/' + dataName + '_' + str(rate) + '_d=' + str(
        d) + '.mat';
    targetPath = 'C:/Users/yifan/OneDrive - The University of Texas at Dallas/Workspace/UTD00_dataminingLab/Code/Data/' + dataName + '/ori/' + dataName + '_' + str(
        rate) + '_target_stream.txt'

    data = sio.loadmat(sfPath)
    # Matrix = sio.loadmat(Lpath)

    tranSourceFeature = data['tranSourceF']
    sourceLabel = data['sourceLabel']
    tranTargetFeature = data['trantargetF']
    targetLabel = data['targetLabel']
    transferMatrix = data['L']



    dbscanSourceL = []
    for i in sourceLabel:
        dbscanSourceL.append(i[0])


    #for i in range(1,10,1):
        #print "eps value: ",i*0.1
    print "len of source",len(sourceLabel)
    db = AgglomerativeClustering(n_clusters=noCluster, linkage='average').fit(tranSourceFeature)
#    db = DBSCAN(eps=0.4,metric='euclidean', min_samples=10).fit(tranSourceFeature)
    labels = db.labels_
    print labels
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "num of clusters", n_clusters_

    #print db.components_
    clusId2label = clus_Info(labels,n_clusters_,sourceLabel)
#    print clusId2label
    #
    print "====================================================================="
#    print clus_predict(db, outlierTest, clusId2label)

