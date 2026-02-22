
#! WIP -- don't push to public yet
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset, Dataset
from torchvision.transforms.transforms import Resize

from Plug_and_Play_Attacks.utils.stylegan import create_image

from Plug_and_Play_Attacks.datasets.celeba import CelebA1000
from Plug_and_Play_Attacks.datasets.custom_subset import SingleClassSubset
from Plug_and_Play_Attacks.datasets.facescrub import FaceScrub
from Plug_and_Play_Attacks.datasets.stanford_dogs import StanfordDogs



class DistanceEvaluation():

    def __init__(self, model, img_size:int, train_set:Dataset,
                 seed:int):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.dataset_name = dataset
        self.model = model
        # self.center_crop_size = center_crop_size
        self.img_size = img_size
        self.seed = seed
        # self.train_set = self.prepare_dataset()
        # self.generator = generator
        
        self.train_set = train_set


    def compute_dist(self, imgs, targets, batch_size=64, rtpt=None):
        self.model.eval()
        self.model.to(self.device)
        target_values = set(targets.cpu().tolist())
        smallest_distances = []
        mean_distances_list = [['target', 'mean_dist']]
        for step, target in enumerate(target_values):
            mask = torch.where(targets == target, True, False)
            imgs_masked = imgs[mask]
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)

            target_embeddings = []
            for x, y in DataLoader(target_subset, batch_size):
                with torch.no_grad():
                    x = x.to(self.device)
                    outputs = self.model(x)
                    target_embeddings.append(outputs.cpu())

            attack_embeddings = []
            for img_batch in DataLoader(TensorDataset(imgs_masked),
                                      batch_size,
                                      shuffle=False):
                with torch.no_grad():
                    img_batch = img_batch[0].to(self.device) #! why?
                    # imgs = create_image(w_batch,
                    #                     self.generator,
                    #                     crop_size=self.center_crop_size,
                    #                     resize=(self.img_size, self.img_size),
                    #                     device=self.device,
                    #                     batch_size=batch_size)
                    # imgs = imgs.to(self.device)
                    outputs_batch = self.model(img_batch)
                    attack_embeddings.append(outputs_batch.cpu())

            target_embeddings = torch.cat(target_embeddings, dim=0)
            attack_embeddings = torch.cat(attack_embeddings, dim=0)
            distances = torch.cdist(attack_embeddings, target_embeddings,
                                    p=2).cpu()
            distances = distances**2
            distances, _ = torch.min(distances, dim=1)
            smallest_distances.append(distances.cpu())
            mean_distances_list.append([target, distances.cpu().mean().item()])

            if rtpt:
                rtpt.step(
                    subtitle=
                    f'Distance Evaluation step {step} of {len(target_values)}')

        smallest_distances = torch.cat(smallest_distances, dim=0)
        return smallest_distances.mean(), mean_distances_list

    def find_closest_training_sample(self, imgs, targets, batch_size=64):
        self.model.eval()
        self.model.to(self.device)
        closest_imgs = []
        smallest_distances = []
        resize = Resize((self.img_size, self.img_size), antialias=True)
        for img, target in zip(imgs, targets):
            img = img.to(self.device)
            img = resize(img)
            if torch.is_tensor(target):
                target = target.cpu().item()
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)
            if len(img) == 3:
                img = img.unsqueeze(0)
            target_embeddings = []
            with torch.no_grad():
                # Compute embedding for generated image
                output_img = self.model(img).cpu()
                # Compute embeddings for training samples from same class
                for x, y in DataLoader(target_subset, batch_size):
                    x = x.to(self.device)
                    outputs = self.model(x)
                    target_embeddings.append(outputs.cpu())
            # Compute squared L2 distance
            target_embeddings = torch.cat(target_embeddings, dim=0)
            distances = torch.cdist(output_img, target_embeddings, p=2)
            distances = distances**2
            # Take samples with smallest distances
            distance, idx = torch.min(distances, dim=1)
            smallest_distances.append(distance.item())
            closest_imgs.append(target_subset[idx.item()][0])
        return closest_imgs, smallest_distances
