from mil.data.datasets import musk1, musk2, protein, elephant, corel_dogs, mnist_bags

(bags_train, y_train), (bags_test, y_test) = corel_dogs.load()
print(bags_train.shape)