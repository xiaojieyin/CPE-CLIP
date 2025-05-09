import os
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils.tools import set_random_seed

from data.fewshot_datasets import *
from data.vit_rollout import VITAttentionRollout
import cv2
import random
import torchvision.transforms as transforms

ID_to_DIRNAME = {
    'oxford_flowers': 'Flower102',
    'dtd': 'DTD',
    'oxford_pets': 'OxfordPets',
    'stanford_cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'fgvc_aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'caltech256': 'caltech256',
    'cub': 'cub',
    'birdsnap': 'birdsnap',
    'imagenet': 'imagenet',
    'imagenet_a': 'imagenet-adversarial',
    'imagenetv2': 'imagenetv2',
    'imagenet_r': 'imagenet-rendition',
    'imagenet_sketch': 'imagenet-sketch',
}


def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id in ['imagenet', 'imagenet_a', 'imagenet_sketch', 'imagenet_r', 'imagenetv2']:
        if set_id == 'imagenet':
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'images', 'val')
        elif set_id == 'imagenetv2':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenetv2-matched-frequency-format-val')
        elif set_id == 'imagenet_a':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-a')
        elif set_id == 'imagenet_r':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-r')
        elif set_id == 'imagenet_sketch':
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform,
                                            mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform,
                                            mode=mode)
    elif set_id in ['imagenet_lt']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        testset = ImageNet_LT(testdir, transform=transform, mode='train')
    elif set_id in ['sun_lt']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = SUN_LT(testdir, transform=transform, mode='train')
    else:
        raise NotImplementedError

    return testset


# Transforms
def get_preaugment():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])


def aug(image, preprocess):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed


class Augmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [aug(x, self.preprocess) for _ in range(self.n_views)]

        return [image] + views


def get_random_states():
    random_state = random.getstate()
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    return {
        'random': random_state,
        'torch': torch_state,
        'numpy': numpy_state
    }


def set_random_states(states):
    random.setstate(states['random'])
    torch.set_rng_state(states['torch'])
    np.random.set_state(states['numpy'])


# Transforms with mask
def aug_mask(image, mask, preprocess, resolution=224, brightness=False):
    preaugment = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.8, 1.2)) if brightness else transforms.Lambda(lambda x: x)
    ])
    preaugment_mask = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip()
    ])

    initial_states = get_random_states()
    x_orig = preaugment(image)

    set_random_states(initial_states)
    mask_orig = preaugment_mask(mask)

    x_processed = preprocess(x_orig)

    return x_processed, np.mean(mask_orig)


class AttnAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, resolution=224):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.resolution = resolution

        self._init_models()

    def _init_models(self):
        initial_states = get_random_states()
        dino = torch.hub.load('dino/', 'dino_vitb16', source='local').eval()
        self.rollout = VITAttentionRollout(dino, discard_ratio=0.5, head_fusion="max")
        set_random_states(initial_states)

    def __call__(self, x):
        image_tensor = self.preprocess(self.base_transform(x))

        with torch.no_grad():
            mask = self.rollout(image_tensor.unsqueeze(0))
            mask = cv2.resize(mask, (x.size[0], x.size[1]))
            mask = Image.fromarray(mask)

        views_aug = [aug_mask(x, mask, self.preprocess, self.resolution, brightness=True) for _ in range(self.n_views)]
        scores = [x for _, x in views_aug]
        lower, upper = np.mean(scores) - 2 * np.std(scores, ddof=1), np.mean(scores) + 2 * np.std(scores, ddof=1)
        views = [tensor for tensor, value in views_aug if lower <= value <= upper]
        # views = [tensor for tensor, value in views_aug if lower <= value]

        return [image_tensor] + views
