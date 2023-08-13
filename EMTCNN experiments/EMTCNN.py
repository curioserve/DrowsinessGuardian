from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import tensorflow as tf

def model() : 

	Input = keras.Input(shape=(175, 175, 3))
	conv1 = layers.Conv2D(filters=48, kernel_size=3, strides=1, padding='same')(Input)
	BN1 = layers.BatchNormalization()(conv1)
	MP1 = layers.MaxPool2D(pool_size=(3, 3), strides=2)(BN1)
	conv2 = layers.Conv2D(filters=56, kernel_size=3, strides=1, padding='same')(MP1)
	BN2 = layers.BatchNormalization()(conv2)
	MP2 = layers.MaxPool2D(pool_size=(3, 3), strides=2)(BN2)

	MP3 = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(MP2)
	conv4 = layers.Conv2D(filters=56, kernel_size=1, strides=1, padding='same')(MP3)
	conv5 = layers.Conv2D(filters=56, kernel_size=1, strides=1, padding='same')(MP2)
	conv6 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(MP2)
	conv7 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(MP2)
	conv8 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same')(conv5)
	conv9 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(conv6)
	combined = layers.concatenate([conv9, conv8, conv7, conv4])
	MP4 = layers.MaxPool2D(pool_size=(3, 3), strides=2)(combined)

	# Downsample MP4 to match spatial dimensions of MP5_input
	MP4_downsampled = layers.Conv2D(filters=72, kernel_size=3, strides=2)(MP4)

	conv10 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(MP4)
	conv11 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(conv10)
	conv12 = layers.Conv2D(filters=72, kernel_size=3, strides=1, padding='same')(conv11)
	MP5_input = layers.MaxPool2D(pool_size=(3, 3), strides=2)(conv12)

	# Adjust the spatial dimensions of MP5_input to match MP4_downsampled
	#MP5_input_adjusted = layers.Conv2D(filters=248, kernel_size=1, strides=1, padding='same')(MP5_input)

	# Residual block
	residual = layers.Add()([MP4_downsampled, MP5_input])
	residual = layers.Activation('relu')(residual)

	flatten = layers.Flatten()(residual)
	Dense1 = layers.Dense(128, activation='relu')(flatten)
	Dense2 = layers.Dense(6, activation='relu')(Dense1)
	Dense3 = layers.Dense(4, activation='softmax')(Dense2)

# Create the model


	# Create the model
	model = keras.Model(Input,Dense3)
	model.compile(loss= 'categorical_crossentropy',optimizer=tf.optimizers.Adam(learning_rate=0.0001) ,metrics= ['accuracy'])
	return model

class EMTCNN() : 
	def __init__(self):
		self.model = model()
	
	def load_weights(self,weights_path) : 
		self.model.load_weights(path)
		
	def preprocess_input(self,img) :
		return np.array(img.resize((175,175),resample=Image.BILINEAR))[:,:,:3].reshape((1,175,175,3))/255
		
			
	def predict_single_image(self,img) :
		img = self.preprocess_input(img)
		prediction = np.array(self.model.predict(img)[0])
		max_pred = np.argmax(prediction)
		confidence = prediction[max_pred]
		if max_pred == 0 : 
			return ('eye open',confidence)
		elif max_pred == 1 :
			return ('eye close',confidence)
		if max_pred == 2 : 
			return ('yawn',confidence)
		else : 
			return ('no yawn',confidence)
	def predict_mouth_state(self,img) :

		img = self.preprocess_input(img)
		prediction = np.array(self.model.predict(img)[0])
		max_pred = np.argmax(prediction[2:])
		confidence = prediction[max_pred]
		if max_pred == 0 : 
			return ('yawn',confidence)
		else : 
			return ('no yawn',confidence)
	 
	def predict_eye_state(self,img):
		img = self.preprocess_input(img)
		prediction = np.array(self.model.predict(img)[0])
		max_pred = np.argmax(prediction[:2])
		confidence = prediction[max_pred]
		if max_pred == 0 : 
			return ('eye open',confidence)
		elif max_pred == 1 :
			return ('eye close',confidence)

		 
		 
		
	


