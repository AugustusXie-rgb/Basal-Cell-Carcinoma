import pickle
import matplotlib.pyplot as plt

with open('/media/xiejun/data1/keras_resnet/trainhistory/NSC_group32_1cut_0.txt','rb') as file_pi:
    history = pickle.load(file_pi)

epochs = range(len(history['accuracy']))
plt.figure()
plt.plot(epochs, history['accuracy'], 'b', label='Train acc')
plt.plot(epochs, history['val_accuracy'], 'r', label='Val acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig('/media/xiejun/data1/keras_resnet/trainhistory/temp_acc.jpg')

plt.figure()
plt.plot(epochs, history['loss'], 'b', label='Train loss')
plt.plot(epochs, history['val_loss'], 'r', label='Val loss')
plt.legend()
plt.savefig('/media/xiejun/data1/keras_resnet/trainhistory/temp_loss.jpg')
