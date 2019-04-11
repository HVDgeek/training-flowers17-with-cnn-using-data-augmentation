import warnings
warnings.simplefilter('ignore')
from hvdev.preprocessing import ImageToArrayPreprocessor
from hvdev.preprocessing import AspectAwerePreprocessor
from hvdev.datasets import SimpleDatasetLoader
from hvdev.nn.cnn import MiniVggNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import imutils
import os 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = "path to the image")
ap.add_argument('-o', '--output', required = True , help = "path to save output plot")
args =  vars(ap.parse_args())

print('[INFO] loading dataset...')
imagePaths = list(paths.list_images(args['dataset']))
classeNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classeNames = [str(x) for x in np.unique(classeNames)]

aap = AspectAwerePreprocessor(64, 64)
iam = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors = [aap , iam])

(data , labels ) = sdl.load(imagePaths , verbose = 500)

data = data.astype('float32')/255.0
print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes/(1024*1024.0)))

(trainX, testX , trainY, testY) = train_test_split(data , labels, test_size = 0.25, 
    random_state = 42)

testY = LabelBinarizer().fit_transform(testY)
trainY = LabelBinarizer().fit_transform(trainY)

print('[INFO] compiling model...')
model = MiniVggNet().build(width = 64, height = 64, depth = 3, classes = len(classeNames))
opt = SGD(lr = 0.05)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

print('[INFO] training Network...')
H = model.fit(trainX , trainY , validation_data = (testX, testY), batch_size = 32, 
    epochs = 100 , verbose = 1)

print('[INFO] Evaluating model...')
predictions = model.predict(testX , batch_size = 32).argmax(axis = 1)
print(classification_report(testY.argmax(axis = 1), predictions, target_names = classeNames))

print('[INFO] ploting...')
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0 , 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0 , 100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0 , 100), H.history['acc'], label = "train_acc")
plt.plot(np.arange(0 , 100), H.history['val_acc'], label = 'val_acc')
plt.title('Training Loss and Acurracy')
plt.xlabel('#epochs')
plt.ylabel('loss/acc')
plt.legend()
plt.savefig(args['output'])
plt.show()