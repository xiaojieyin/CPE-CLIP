import argparse
import time
import os
import json
import pickle
import re
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed, accuracy_multi_hot
from data.cls_to_names import get_classnames, CUSTOM_TEMPLATES
from data.datautils import Augmenter, build_dataset, AttnAugmenter

from clip import clip


def load_clip_to_cpu(arch):
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


@torch.no_grad()
def clip_evaluation(val_loader, clip_model, args):
    dataset_name = args.test_set
    print("Evaluating: {}".format(dataset_name))
    classnames = get_classnames(dataset_name)

    # ============== prepare text features ==============
    prompts = []
    template = "a photo of a {}."
    for classname in classnames:
        prompt = template.format(classname.replace("_", " "))
        prompts.append(clip.tokenize(prompt))
    prompts = torch.cat(prompts).cuda()

    with torch.amp.autocast('cuda'):
        text_features = clip_model.encode_text(prompts)  # n_cls x d
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ==============  calculate logits for each image ==============
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    print("number of test samples: {}".format(len(val_loader)))
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    end = time.time()
    for i, (images, target) in enumerate(tqdm(val_loader, total=len(val_loader))):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        output = clip_model.logit_scale.exp() * image_features @ text_features.t()

        # measure accuracy
        acc1, = accuracy(output, target, topk=(1,))
        top1.update(acc1[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    print(f'\n *  {dataset_name}')
    progress.display_summary()
    return [top1.avg, batch_time.avg]


def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model = load_clip_to_cpu(args.arch)
    clip_model = clip_model.cuda()
    clip_model.float()
    clip_model.eval()

    for _, param in clip_model.named_parameters():
        param.requires_grad_(False)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    data_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = build_dataset(args.test_set, data_transform, args.data, mode='test')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # testing start
    return clip_evaluation(val_loader, clip_model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWT evaluation')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('--data', type=str, help='data path')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--descriptor_path', type=str)
    parser.add_argument('--num_descriptor', type=int, default=50)

    parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N')
    parser.add_argument('--resolution', default=224, type=int, help='image resolution')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    set_random_seed(args.seed)
    args.device = 'cuda:0'

    datasets = args.test_set.split("/")

    results = {}
    for set_id in datasets:
        args.test_set = set_id
        results[set_id] = main_worker(args)

        print("======== Result Summary ========")
        print("\t\t [set_id] \t\t Top-1 acc. \t\t time.")
        for id in results.keys():
            print("{}".format(id), end="	")
        print("\n")
        for id in results.keys():
            print("{:.2f}".format(results[id][0]), end="	")
        print("\n")
        for id in results.keys():
            print("{:.4f}".format(results[id][1]), end="	")
        print("\n")

