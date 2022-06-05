from CNN_model import build_model
from get_data_splits import get_data_splits
import matplotlib.pyplot as plt

def train_model(data_path, learning_rate, epochs, batch_size, saved_model_path):

    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(data_path)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # mfccs, 1)
    model = build_model(input_shape, learning_rate, num_keywords = max(y_test)+1)

    # train the model
    history = model.fit(X_train, y_train, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        validation_data = (X_validation, y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test error: {test_error}, test accuracy: {test_accuracy}')

    # save model
    model.save(saved_model_path)

    return history

# ---

if __name__ == '__main__':

    # Parameters of the train
    data_path = 'flask_api/keyword_spotting_service/data.json'
    learning_rate = 0.0001
    epochs = 100
    batch_size = 32
    saved_model_path = 'flask_api/keyword_spotting_service/CNNmodel.h5'
    
    # Training
    history = train_model(data_path, learning_rate, epochs, batch_size, saved_model_path)

    # Plots of accuracy in test and validation
    fig_acc = plt.figure(dpi = 500)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.ylim(0,1)
    plt.grid()
    plt.savefig('images/model_accuracy.png')
    plt.show()

    # Plots of loss in test and validation
    fig_loss = plt.figure(dpi = 500)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid()
    plt.savefig('images/model_loss.png')
    plt.show()