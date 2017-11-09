import cv2
import os
import random
import numpy as np
from tqdm import tqdm
np.set_printoptions(precision=2)

import openface

##################
# docker run command:
# docker run --privileged -p 9007:9007 -p 8007:8007 -t -v C:/Users/IASA-FRI/Desktop/arface_mount:/root/openface/demos/arface -i bamos/openface /bin/bash
##############

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..','..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

arfaceDir = os.path.join(fileDir,'arface_png')

#Path to dlib's face predictor
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
#Path to Torch network model
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
#input image dimension
imgDim = 96

align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, imgDim)

numSamePairs = 3000
numDiffPairs = 3000

###############################
#generate image pairs
def genPairs(imgDir,numSamePairs,numDiffPairs): 
    allimgs = os.listdir(imgDir)
    goodimgs = []
    for img in tqdm(allimgs): 
        imgPath = os.path.join(imgDir,img)
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        bb = align.getLargestFaceBoundingBox(rgbImg)
        if bb is not None:
            goodimgs.append(img)
    people = list(set(map(lambda x: '-'.join(x.split("-")[0:2]),goodimgs)))
    imgsByPerson = []
    for person in tqdm(people):
        personImgs = []
        for img in goodimgs:
            if person == '-'.join(img.split("-")[0:2]): 
                personImgs.append(img)
        imgsByPerson.append(personImgs)

    samePeoplePairs = []
    numPeople = len(imgsByPerson)
    for i in tqdm(range(numSamePairs)):
        samePeoplePairs.append((random.choice(imgsByPerson[i%numPeople]),random.choice(imgsByPerson[i%numPeople])))

    diffPeoplePairs = []
    for i in tqdm(range(numDiffPairs)):
        j = random.choice([p for p in range(numPeople) if p != i])
        diffPeoplePairs.append((random.choice(imgsByPerson[i%numPeople]),random.choice(imgsByPerson[j%numPeople])))
    
    return [samePeoplePairs,diffPeoplePairs]

# list where [0] is samePeoplePairs and [1] is diffPeoplePairs
peoplePairs = genPairs(arfaceDir,numSamePairs,numDiffPairs)


def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    rep = net.forward(alignedFace)
    # print("Representation:")
    # print(rep)
    # print("-----\n")
    return rep

def l2dist(img1,img2):
    d = getRep(img1) - getRep(img2)
    return np.dot(d, d)

correctGuesses = 0
for (img1, img2) in tqdm(peoplePairs[0]):
    if l2dist(os.path.join(arfaceDir,img1),os.path.join(arfaceDir,img2)) < 1:
        correctGuesses += 1 
accuracySamePeople = correctGuesses / float(numSamePairs)

print "correctGuesses for samePairs",correctGuesses

correctGuesses = 0
for (img1, img2) in tqdm(peoplePairs[1]):
    if l2dist(os.path.join(arfaceDir,img1),os.path.join(arfaceDir,img2)) >= 1:
        correctGuesses += 1 
accuracyDiffPeople = correctGuesses / float(numDiffPairs)

print "correctGuesses for diffPairs",correctGuesses
print "Accuracy for same people: ",accuracySamePeople
print "Accuracy for different people: ",accuracyDiffPeople
