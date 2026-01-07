import numpy as np
import os

# ==========================================
# 🛠️ MONKEY PATCH: FIX DEPENDENCY HELL
# ==========================================
# Numpy 1.20+ removed np.object and np.bool, but tensorflowjs 3.x needs them.
# We manually restore them so the library doesn't crash.
try:
    np.object = object
    np.bool = bool
except AttributeError:
    pass

# Now it is safe to import tensorflowjs
import tensorflowjs as tfjs
import tensorflow as tf

# ==========================================
# 🔄 CONVERSION LOGIC
# ==========================================
input_path = 'sheep_index_model.h5'
output_folder = './sheep_scanner/model'

# 1. Check if model exists
if not os.path.exists(input_path):
    print(f"❌ Error: Could not find {input_path}")
    print("Make sure the .h5 file is in the same folder as this script!")
    exit()

# 2. Create output directory
os.makedirs(output_folder, exist_ok=True)

print(f"📂 Loading {input_path}...")

# 3. Load the model
try:
    model = tf.keras.models.load_model(input_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Keras model: {e}")
    exit()

print("🚀 Converting to TensorFlow.js format...")

# 4. Convert
# We explicitly specify the input format to ensure compatibility
try:
    tfjs.converters.save_keras_model(model, output_folder)
    print(f"🎉 Success! Files saved to: {output_folder}")
    print("You should see 'model.json' and 'group1-shard1of1.bin' in that folder.")
except Exception as e:
    print(f"❌ Conversion failed: {e}")