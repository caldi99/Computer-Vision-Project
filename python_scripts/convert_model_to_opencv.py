"""
    This file was entirely created by Francesco Caldivezzi
"""

#TENSORFLOW LIB
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#OTHER LIBS
import numpy as np

#MYLIBS
from config import configs

#Get the model
model = keras.models.load_model(configs.MODEL_PATH)

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(x = tf.TensorSpec(model.inputs[0].shape, 
                                                                model.inputs[0].dtype))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

#Inspect the layers operations inside your frozen graph definition and see the name of its input and output tensors
layers = [op.name for op in frozen_func.graph.get_operations()]

print("FRONZEN LAYERS : ")
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive serialize the frozen graph and its text representation to disk.
tf.io.write_graph(graph_or_graph_def = frozen_func.graph,
                  logdir = configs.MODEL_PATH_PB_DIR,
                  name= configs.MODEL_PATH_PB,
                  as_text=False)

'''#Optional
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pbtxt",
                as_text=True)'''

print("MODEL SUMMARY : ")
model.summary()