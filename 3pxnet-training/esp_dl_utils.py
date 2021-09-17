import tensorflow as tf
import numpy as np

"""
esp-dl represents quantized weights as a matrix of uint8 values * 2^exponent
The goal of this getExponent is to return the largest integer exponent e possible such that, weight*2^e fits between [-128, 127]
"""
def getExponent(weight):
  weight_sorted = tf.sort(tf.reshape(weight, [-1]))
  weight_min = weight_sorted[int((len(weight_sorted)-1)*0.00)]
  weight_max = weight_sorted[int((len(weight_sorted)-1)*1.00)]
  weight = tf.clip_by_value(weight, clip_value_min=weight_min, clip_value_max=weight_max)
  
  #scaling_factor = 128/tf.maximum(tf.reduce_max(weight), tf.reduce_max(-weight))
  scaling_factor = 32768/tf.maximum(weight_max, -weight_min) #2^16/2
  #this is expected to happen when values in weight are very small (so scaling factor is very large)
  if tf.math.is_nan(scaling_factor) or tf.math.is_inf(scaling_factor):
    return tf.constant(-10.)
  e = tf.math.ceil(-tf.math.log(scaling_factor)/tf.math.log(2.))
  return e

def getQuantizedWeight(weight):
  #fig, axs = plt.subplots(1, 2)
  #weight_sorted = tf.sort(tf.reshape(weight, [-1]))
  #axs[0].hist(weight_sorted)
  
  e = getExponent(weight)
  two_pow_e = tf.math.pow(2, e)
  weight_quantized = weight / two_pow_e
  weight_quantized = tf.cast(tf.math.round(weight_quantized), tf.int16)

  #weight_quantized_sorted = tf.sort(tf.reshape(tf.math.multiply(tf.cast(weight_quantized, tf.float32), two_pow_e), [-1]))
  #axs[1].hist(weight_quantized_sorted)

  return e, weight_quantized

"""
The goal of this function is to get the floating point approximation whose values are equal to the representation created by the int8 quantization
"""
def getApproximateQuantizedWeight(weight):
  e, weight_quantized = getQuantizedWeight(weight)
  two_pow_e = tf.math.pow(2, e)
  weight_quantized_approximation = tf.math.multiply(tf.cast(weight_quantized, tf.float32), two_pow_e)
  return e, weight_quantized_approximation


def writeFullPrecWeights(weights, outputFileName="model.h"):
  all_code = ""
  all_code += "#pragma once\n"
  all_code += "#include \"dl_lib_matrix3d.h\"\n\n"
  #all_code += "dl_matrix3d_t* model_forward(dl_matrix3d_t *input);\n\n"

  # data ordering should be NWHC for conv, NHWC for dense
  for (weight_name, weight_value) in weights.items():
    weight_arr = weight_value["value"].detach().cpu().numpy()
    if weight_value["type"] == "conv_kernel":
      #https://github.com/espressif/esp-dl/blob/master/lib/include/dl_lib_matrix3d.h#L667
      #goal: convert nchw (torch) -> nwhc (esp-dl)
      weight_arr = np.transpose(weight_arr, (0, 3, 2, 1))
      n, w, h, c = weight_arr.shape
    elif weight_value["type"] == "conv_bias":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    elif weight_value["type"] == "dense_kernel":
      #https://stackoverflow.com/questions/50212330/dense-layer-weights-shape
      #https://github.com/espressif/esp-dl/blob/master/lib/include/dl_lib_matrix3d.h#L770
      #goal: convert (output_no, input_no) -> (n, w, h, c)=(1, input_no, output_no, 1)
      n, w, h, c = 1, weight_arr.shape[1], weight_arr.shape[0], 1
    elif weight_value["type"] == "dense_bias":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    elif weight_value["type"] == "batchnorm_scale":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    elif weight_value["type"] == "batchnorm_offset":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    stride = w*c

    matrix_data_flat = weight_arr.flatten().tolist()

    matrix_data_code = "const static fptp_t " + weight_name + "_item_array[] = {\n"
    for i in range(0, len(matrix_data_flat), 8):
      matrix_data_code += "\t"
      for j in range(8):
        if i+j >= len(matrix_data_flat):
          break
        matrix_data_code += "%.5f" % matrix_data_flat[i+j] + "f"
        if not i+j==len(matrix_data_flat)-1:
          matrix_data_code += ", "
      if not i+8 >= len(matrix_data_flat):
        matrix_data_code += "\n"
    matrix_data_code += "\n};"

    matrix_code = "static dl_matrix3d_t " + weight_name + " = {"
    matrix_code += "\n\t.w = " + str(w) + ","
    matrix_code += "\n\t.h = " + str(h) + ","
    matrix_code += "\n\t.c = " + str(c) + ","
    matrix_code += "\n\t.n = " + str(n) + ","
    matrix_code += "\n\t.stride = " + str(stride) + ","
    matrix_code += "\n\t.item = (fptp_t *)(&" + weight_name + "_item_array[0])"
    matrix_code += "\n};"


    all_code += matrix_data_code + "\n\n"
    all_code += matrix_code + "\n\n"

  print("writing weights to", outputFileName)
  with open(outputFileName, "w") as f:
    f.write(all_code)

def writeQuantizedWeights(weights, outputFileName="model_qu.h"):
  all_code = ""
  all_code += "#pragma once\n"
  all_code += "#include \"dl_lib_matrix3dq.h\"\n\n"
  #all_code += "dl_matrix3d_t* model_forward(dl_matrix3du_t *input);\n\n"

  # data ordering should be NWHC for conv, NHWC for dense
  for (weight_name, weight_value) in weights.items():
    weight_arr = weight_value["value"].detach().cpu().numpy()
    if weight_value["type"] == "conv_kernel":
      #https://github.com/espressif/esp-dl/blob/master/lib/include/dl_lib_matrix3d.h#L667
      #goal: convert nchw (torch) -> nwhc (esp-dl)
      weight_arr = np.transpose(weight_arr, (0, 3, 2, 1))
      n, w, h, c = weight_arr.shape
    elif weight_value["type"] == "conv_bias":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    elif weight_value["type"] == "dense_kernel":
      #https://stackoverflow.com/questions/50212330/dense-layer-weights-shape
      #https://github.com/espressif/esp-dl/blob/master/lib/include/dl_lib_matrix3d.h#L770
      #goal: convert (output_no, input_no) -> (n, w, h, c)=(1, input_no, output_no, 1)
      n, w, h, c = 1, weight_arr.shape[1], weight_arr.shape[0], 1
    elif weight_value["type"] == "dense_bias":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    elif weight_value["type"] == "batchnorm_scale":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    elif weight_value["type"] == "batchnorm_offset":
      #goal: convert (output_no) -> (n, w, h, c)=(1, 1, 1, output_no)
      n, w, h, c = 1, 1, 1, weight_value["value"].shape[0]
    stride = w*c

    matrix_data_flat_float = weight_arr.flatten()
    
    e, matrix_data_flat = getQuantizedWeight(matrix_data_flat_float)
    matrix_data_flat = matrix_data_flat.numpy().tolist()

    matrix_data_code = "const static qtp_t " + weight_name + "_item_array[] = {\n"
    for i in range(0, len(matrix_data_flat), 8):
      matrix_data_code += "\t"
      for j in range(8):
        if i+j >= len(matrix_data_flat):
          break
        matrix_data_code += "%d" % matrix_data_flat[i+j]
        if not i+j==len(matrix_data_flat)-1:
          matrix_data_code += ", "
      if not i+8 >= len(matrix_data_flat):
        matrix_data_code += "\n"
    matrix_data_code += "\n};"

    matrix_code = "static dl_matrix3dq_t " + weight_name + " = {"
    matrix_code += "\n\t.w = " + str(w) + ","
    matrix_code += "\n\t.h = " + str(h) + ","
    matrix_code += "\n\t.c = " + str(c) + ","
    matrix_code += "\n\t.n = " + str(n) + ","
    matrix_code += "\n\t.stride = " + str(stride) + ","
    matrix_code += "\n\t.exponent = " + str(int(e)) + ","
    matrix_code += "\n\t.item = (qtp_t *)(&" + weight_name + "_item_array[0])"
    matrix_code += "\n};"

    all_code += matrix_data_code + "\n\n"
    all_code += matrix_code + "\n\n"

  print("writing weights to", outputFileName)
  with open(outputFileName, "w") as f:
    f.write(all_code)
