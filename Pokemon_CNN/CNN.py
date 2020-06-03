from keras import preprocessing
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


with open("data\\label_pokm.json", "r") as f:
    labeled_text = json.load(f)

# get a list_label and list_dataset for CNN
# there are too many noises from type "poison"
label = ["Fire", "Water", "Grass"]

"""
label_data = []
data = []
for typ in label:
    for pokm in labeled_text[typ]:
        for num in pokm:
            filename = num + pokm[num][0] + ".png"
            dire = typ.lower()
            pokm_img = Image.open(f"data\\{dire}\\{filename}")
            (width, height) = (pokm_img.width // 10, pokm_img.height // 10)
            pokm_img = pokm_img.resize((width, height))
            pokm_arr = np.array(pokm_img)
            pokm_arr = np.divide(pokm_arr, 255)
            label_data.append(typ)
            data.append(pokm_arr)



# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(60, 60, 4), activation= "relu"))
model.add(MaxPooling2D(pool_size= (3, 3)))
model.add(Dropout(0.4))

# Second convolutional layer
model.add(Conv2D(8, (3, 3), activation= "relu"))
model.add(MaxPooling2D(pool_size= (3, 3)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(units=512, activation= "relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation= 'relu'))
model.add(Dense(units=len(label), activation= 'softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

# generate x_data, y_data
x = data


y = [[0 for i in range(len(label))] for item in label_data]
for typ in label:
    for num, typ2 in enumerate(label_data):
        if typ == typ2:
            y[num][label.index(typ)] = 1

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=1)


# fit the model
type_model = model.fit(x_train, y_train, batch_size=16, epochs= 100, verbose = 2, validation_data= (x_test, y_test), shuffle= True)

# summarize history for accuracy
plt.plot(type_model.history['accuracy'])
plt.plot(type_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("CNN_model.h5")
"""
#get around 75~80% accuracy for validation set, share your parameter if you have better results :)

#model = load_model("CNN_model.h5")
# This should be the best model so far
model = load_model("CNN_bestModelSoFar.h5")

# make prediction
Palkia = "484Palkia"
Simisear = "514Simisear"
Ferrothorn = "598Ferrothorn"
Frogadier = "657Frogadier"
Talonflame = "663Talonflame"
predict = Frogadier
path = f"test_image\\{predict}.png"
pokm_img = Image.open(path)
pokm_img = pokm_img.resize((60, 60))
pokm_arr = np.array(pokm_img, dtype = float)

pokm_arr = pokm_arr/ 255

# Idk why it need a fourth dimension
pokm_arr = pokm_arr.reshape(-1, 60, 60, 4)

result = model.predict(pokm_arr)
print(f"Predicted type for {predict}")
result_dict = dict(zip(label, result.tolist()[0]))
print(label[np.argmax(result)])
print(result_dict)

# comment: it seems like the model prioritise shape over color
# it will be better if we just use the type-1 pokemon in training
# Probabilty better if there are more data :P