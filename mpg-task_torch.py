import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.models import resnet101, ResNet101_Weights, ResNeXt101_32X8D_Weights, resnext101_32x8d, resnet50, ResNet50_Weights

from sklearn.decomposition import PCA

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
plt.style.use('seaborn-whitegrid')


######################
# modify
modelname = 'resnext101'                 # resnet50 / resnet101 / resnext101


#######################
dir = os.getcwd()
files = os.path.join(dir, 'val')
activations = os.path.join(dir, 'activations')
if not os.path.exists(files):
    os.makedirs(files)
if not os.path.exists(activations):
    os.makedirs(activations)


if modelname == 'resnet101':
    model_activation = os.path.join(activations, modelname)
    if not os.path.exists(model_activation):
        os.makedirs(model_activation)
    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights)
    # model.eval()

if modelname == 'resnext101':
    model_activation = os.path.join(activations, modelname)
    if not os.path.exists(model_activation):
        os.makedirs(model_activation)
    weights = ResNeXt101_32X8D_Weights.DEFAULT
    model = resnext101_32x8d(weights=weights)
    # model.eval()

if modelname == 'resnet50':
    model_activation = os.path.join(activations, modelname)
    if not os.path.exists(model_activation):
        os.makedirs(model_activation)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    # model.eval()


def extract_activation(weights, model):

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.squeeze(0).detach()
        return hook

    preprocessing = weights.transforms()
    activation = {}
    model.layer4.register_forward_hook(get_activation('layer4'))
    count = 0
    pca = PCA(n_components=2)

    for folder in os.listdir(files):
        if not folder.startswith('.'):
            for f in os.listdir(os.path.join(files, folder)):
                count += 1
                if not os.path.isfile(os.path.join(model_activation, f)):
                    if f.endswith('.JPEG'):
                        file = os.path.join(files, folder, f)

                        batch = preprocessing(Image.open(file).convert('RGB')).unsqueeze(0)
                        model(batch)
                        dimensionality_reduction(pca, activation['layer4'].cpu().numpy(), f)
                        print('image: ' + str(count))
    return None


def dimensionality_reduction(pca, activation, image_name):
    x = pca.fit_transform(np.array([activation[j].flatten() for j in range(len(activation))]))
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel('PC-1'), plt.ylabel('PC-2')
    plt.savefig(os.path.join(model_activation, image_name))
    plt.close()
    return None


if __name__ == '__main__':
    extract_activation(weights, model)
