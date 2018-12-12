import os
import cv2
import tensorflow as tf
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from transfLearning import inception_knn
from transfLearning import cluster_vectors

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_path", help="path to images")
ap.add_argument("-m", "--model", help="model type", default="InceptionV4KNN")
ap.add_argument("-p", "--pb", help='path to protobuffer file')

args = vars(ap.parse_args())

products = {'bread': 0, 'sadochok': 1}

models = {"InceptionV4KNN": inception_knn}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main():
	images = glob.glob1(args['images_path'], "*.jpg")
	y_test = [products[i.split("_")[0]] for i in images]
	y_pred = []
	print(y_test)

	model = models[args['model']]

	ann_tree = cluster_vectors.build_ann_index()

	session = tf.Session()
	with session.as_default():
		model.create_graph(args['pb'])

	for i in images:
		image_path = os.path.join(args['images_path'], i)
		img = cv2.imread(image_path)
		img = cv2.resize(img, (299, 299))

		feature_vector = model.run_inference_on_image(session, img)
		print(feature_vector.shape)

		nearest_neighbors = cluster_vectors.nearest_neighbors(ann_tree, feature_vector)
			
		rec_prod = sorted(nearest_neighbors, key=lambda k: k['similarity'])[0]
		rec_class = rec_prod['filename'].split('\\')[-1].split('_')[0]

		y_pred.append(products[rec_class])

	cnf_matrix = confusion_matrix(y_test, y_pred)
	acc = sum([i for i, j in zip(y_pred, y_test) if i == j]) / len(images)
	print("Accuracy: %s" % acc)
	# print(cnf_matrix)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=products,
                      title='Confusion matrix, without normalization')

	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=products, normalize=True,
                      title='Normalized confusion matrix')


	plt.show()




if __name__ == "__main__":
	main()
