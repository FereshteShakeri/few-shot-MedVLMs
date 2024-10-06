from PIL import Image
import numpy as np

# Import FLAIR
from flair import FLAIRModel

# Set model
model = FLAIRModel(from_checkpoint=True)

# Load image and set target categories 
# (if the repo is not cloned, download the image and change the path!)

image = np.array(Image.open("./documents/sample_macular_hole.png"))
text = ["normal", "healthy", "macular edema", "diabetic retinopathy", "glaucoma", "macular hole",
        "lesion", "lesion in the macula"]

# Forward FLAIR model to compute similarities
probs, logits = model(image, text)

print("Image-Text similarities:")
print(logits.round(3)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
print("Probabilities:")
print(probs.round(3))  # [[0.      0.     0.001  0.002  0.01   0.02   0.019  0.948]]

#######################
# compute image features and text features
image_features, text_features = model.flair_classifier(image, text)
print('image_features is ',image_features.shape)
print('text_features is ',text_features.shape)