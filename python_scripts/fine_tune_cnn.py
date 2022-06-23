"""
    This file was entirely created by Francesco Caldivezzi
    In order to execute this file the directory structure must be the following :    
        dataset/
            train/
                hand/
                no_hand/
            validation/
                hand/
                no_hand/
            test/
                hand/
                no_hand/
        configs/
            config.py        
        fine_tune_cnn.py
        test_model.py
"""

#MY IMPORTS
from config import configs

#KERAS IMPORTS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#OTHER LIBS IMPORTS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# PREPARE GENERATORS FOR READING THE DATA
# --------------------------------------------------

#Defintion of datagenerator
train_datagen = ImageDataGenerator(rescale = configs.DATAGENERATOR_RESCALE, #rescale
                                rotation_range = configs.DATAGENERATOR_ROTATIONRANGE, #parameters for data augmentation 
                                zoom_range = configs.DATAGENERATOR_ZOOMRANGE, 
                                width_shift_range = configs.DATAGENERATOR_WIDTHSHIFTRANGE, 
                                height_shift_range = configs.DATAGENERATOR_HEIGTHSHIFTRANGE, 
                                shear_range = configs.DATAGENERATOR_SHEARRANGE,	
                                horizontal_flip = True,	
                                fill_mode="nearest")
validation_datagen =  ImageDataGenerator(rescale = configs.DATAGENERATOR_RESCALE)

#Definition of generators for training validating and testing the NN without loading the entire dataset in RAM
train_generator = train_datagen.flow_from_directory(
    directory = configs.TRAIN_PATH,
    target_size = configs.INPUT_DIMS,
    color_mode = "rgb",
    batch_size = configs.BS,
    class_mode='binary',
    shuffle = True
)
validation_generator = validation_datagen.flow_from_directory(
    directory = configs.VALIDATION_PATH,
    target_size = configs.INPUT_DIMS,
    color_mode = "rgb",
    batch_size = configs.BS,
    class_mode='binary',
    shuffle = True
)
test_generator = validation_datagen.flow_from_directory(
    directory = configs.TEST_PATH,
    target_size = configs.INPUT_DIMS,
    color_mode = "rgb",
    batch_size = 1,
    class_mode = None,
    shuffle = False
)

# --------------------------------------------------
# DEFINTION OF MODEL
# --------------------------------------------------

print("DEFINE MODEL")

#Define The Model
baseModel = ResNet50V2(weights = "imagenet", 
                        include_top = False, 
                        input_tensor = Input(shape = configs.INPUT_TENSOR_DIMS))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

#Make the layers of the ResNet50V2 model not trainable
for layer in baseModel.layers:
	layer.trainable = False

#Compile the Model
opt = Adam(learning_rate = configs.INIT_LR)
model.compile(loss = "binary_crossentropy", 
                optimizer = opt, 
                metrics = ["accuracy"])

# --------------------------------------------------
# TRAIN OF THE MODEL
# --------------------------------------------------

print("START TRAINING")

#Train the model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
H = model.fit(train_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = validation_generator,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = configs.EPOCHS
)

# --------------------------------------------------
# SAVE THE MODEL
# --------------------------------------------------

print("SAVING MODEL IN THE .H5 FORMAT")
model.save(configs.MODEL_PATH, save_format="h5")

# --------------------------------------------------
# TESTING PERFORMANCES ON TEST SET
# --------------------------------------------------

print("TESTING PERFORMANCES ON TEST SET")
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()
predictions = model.predict(test_generator, 
                            steps = STEP_SIZE_TEST,
                            verbose = 1)

y_pred = np.rint(predictions)
y_true = test_generator.classes
print("CONFUSION MATRIX \n",confusion_matrix(y_true,y_pred))

# --------------------------------------------------
# PLOT ACCURACY, TRAINING AND VALIDATION LOSS
# --------------------------------------------------

print("PLOT GRAPH ACCURACY, TRAINING AND VALIDATION LOSS, AND SAVE IT")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, configs.EPOCHS), 
        H.history["loss"], 
        label="train_loss")
plt.plot(np.arange(0, configs.EPOCHS), 
        H.history["val_loss"], 
        label="val_loss")
plt.plot(np.arange(0, configs.EPOCHS), 
        H.history["accuracy"], 
        label="train_acc")
plt.plot(np.arange(0, configs.EPOCHS), 
        H.history["val_accuracy"], 
        label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(configs.PLOT_PATH)
plt.show()