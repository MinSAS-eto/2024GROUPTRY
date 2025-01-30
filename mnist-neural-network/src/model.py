class MNISTModel:
    def __init__(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, Dropout
        from tensorflow.keras.utils import to_categorical

        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_model(self):
        return self.model