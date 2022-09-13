from collections import Counter
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms as T
from torchvision.utils import make_grid
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def extract_features(model, dataset, batch_size=32):
    """
    Gets the output of a pytorch model given a dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    model.eval()
    with torch.no_grad():
        for image, _ in tqdm(loader, desc='extracting features'):
            output = model.get_feature(image.to(device))
            features.append(output.cpu())
    return torch.cat(features).numpy()


class FoodDataset(Dataset):
    def __init__(self, root, transforms=None, labels=[], limit=None):
        self.root = Path(root)
        self.image_paths = list(Path(root).glob('*.jpg'))
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.labels = labels
        self.transforms = transforms
        self.classes = set([path.parts[-2] for path in self.image_paths])

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index] if self.labels else 0
        image = Image.open(image_path)
        if self.transforms:
            return self.transforms(image), label
        return image, label

    def __len__(self):
        return len(self.image_paths)


transforms = T.Compose([T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor()])

# data
root = 'D:/lifengxin/dog_and_cats/train/'
limit_images = 1000

# clustering
pca_dim = 50

# convnet
batch_size = 64
num_classes = 100
num_epochs = 2

dataset = FoodDataset(root=root, limit=limit_images)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.bock_net = resnet18()
        self.bock_net.fc = nn.Identity()
        self.class_header = nn.Linear(512, num_classes)

    def get_feature(self, images):
        return self.bock_net(images)

    def forward(self, images):
        return self.class_header(self.get_feature(images))


# load resnet and alter last layer
model = Model()
model = model.to(device)

pca = IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
kmeans = MiniBatchKMeans(n_clusters=num_classes, batch_size=512, init_size=3 * num_classes)
optimizer = Adam(model.parameters())


def cluster(pca, kmeans, model, dataset, batch_size, return_features=False):
    features = extract_features(model, dataset, batch_size)
    reduced = pca.fit_transform(features)
    pseudo_labels = list(kmeans.fit_predict(reduced))
    if return_features:
        return pseudo_labels, features
    return pseudo_labels


def train_epoch(model, optimizer, train_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    pbar = tqdm(train_loader)
    model.train()
    for batch, (images, labels) in enumerate(pbar):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.long().to(device)
        out = model(images)
        loss = F.cross_entropy(out, labels)
        total_loss += loss.item()
        pbar.set_description(f'training - loss: {total_loss / (batch + 1)}')
        loss.backward()
        optimizer.step()


raw_dataset = FoodDataset(root=root, transforms=transforms, limit=limit_images)

for i in range(num_epochs):
    pseudo_labels = cluster(pca, kmeans, model, raw_dataset, batch_size)  # generate labels
    labeled_dataset = FoodDataset(root=root, labels=pseudo_labels, transforms=transforms,
                                  limit=limit_images)  # make new dataset with labels matched to images
    train_epoch(model, optimizer, labeled_dataset, batch_size)  # train for one epoch
