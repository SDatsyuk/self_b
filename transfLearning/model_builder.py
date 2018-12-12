import tensorflow as tf

from transfLearning.inception_knn import InceptionV4
from transfLearning.mobilenet import MobilenetV2
from transfLearning.resnet import ResnetV2_101

MODELS = {
			"InceptionV4": InceptionV4,
			"MobilenetV2": MobilenetV2,
			"ResnetV2_101": ResnetV2_101
		}

def build_model(model_type):
	if model_type not in MODELS:
		raise ValueError("Model must be one of declared. %s" % MODELS.keys())
	model = MODELS[model_type]()

	return model
