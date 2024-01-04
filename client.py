from typing import Dict, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np
import matplotlib.pyplot as plt
import sys

from model import load_partition, gan, discriminator, generator

CLIENT_ID = int(sys.argv[1])

(x_train, y_train), (x_test, y_test) = load_partition(CLIENT_ID)

x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(x_train.shape[0], 784)

x_test = (x_test.astype(np.float32) - 127.5) / 127.5
x_test = x_test.reshape(x_test.shape[0], 784)

latent_dim = 100

class MNISTClient(fl.client.NumPyClient):
  def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
    return generator.get_weights()

  def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
    generator.set_weights(parameters)

    server_round = config["current_round"]
    epochs = config["local_epochs"]

    def plot_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
      noise = np.random.normal(0, 1, size=[examples, latent_dim])
      generated_images = generator.predict(noise)
      generated_images = generated_images.reshape(examples, 28, 28)

      plt.figure(figsize=figsize)
      for i in range(generated_images.shape[0]):
          plt.subplot(dim[0], dim[1], i + 1)
          plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
          plt.axis('off')
      plt.tight_layout()
      plt.savefig(f'results/round_{server_round}_client_{CLIENT_ID}_epoch_{epoch}.png')

    def train_gan(epochs=1, batch_size=128):
      batch_count = x_train.shape[0] // batch_size

      for e in range(epochs):
          for _ in range(batch_count):
              noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
              generated_images = generator.predict(noise)
              
              image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

              X = np.concatenate([image_batch, generated_images])
              y_dis = np.zeros(2 * batch_size)
              y_dis[:batch_size] = 0.9  # Rótulos suavizados para o treinamento estável

              # Treina o discriminador
              d_loss = discriminator.train_on_batch(X, y_dis)

              # Calcula a acurácia do discriminador
              d_acc_real = np.mean(discriminator.predict(image_batch) > 0.5)
              d_acc_fake = np.mean(discriminator.predict(generated_images) <= 0.5)
              d_acc = 0.5 * (d_acc_real + d_acc_fake)


              noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
              y_gen = np.ones(batch_size)

              # Treina a GAN (o gerador é treinado via gan)
              g_loss = gan.train_on_batch(noise, y_gen)

          print(f'Época {e+1}/{epochs}, Discriminador Loss: {d_loss}, GAN Loss: {g_loss}')
          print(f'Discriminador Acurácia: {100 * d_acc}%')

          # Salva imagens geradas a cada 10 épocas
          if (e + 1) % 1 == 0:
              plot_generated_images(e, generator)

        # Retorna os pesos do gerador ao final do treinamento
      return generator.get_weights()

    train_gan(epochs=epochs)

    return (generator.get_weights(), len(x_train), {})
  

  def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
    server_round = config["current_round"]

    def plot_generated_images(generator, examples=10, dim=(1, 10), figsize=(10, 1)):
      noise = np.random.normal(0, 1, size=[examples, latent_dim])
      generated_images = generator.predict(noise)
      generated_images = generated_images.reshape(examples, 28, 28)

      plt.figure(figsize=figsize)
      for i in range(generated_images.shape[0]):
          plt.subplot(dim[0], dim[1], i + 1)
          plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
          plt.axis('off')
      plt.tight_layout()
      plt.savefig(f'results/round_{server_round}_client_{CLIENT_ID}_evaluate.png')
      plt.close()

    # generator.set_weights(parameters)
    plot_generated_images(generator)
    
    #TODO: EVALUATE GAN METRICS
    return (0.9, len(x_train), {})
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MNISTClient())