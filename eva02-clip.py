import argparse
from PIL import Image
from tqdm import tqdm
import os
import time
import re

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import Augmenter, build_dataset, AttnAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *
from evaluate import get_entropy_weight, Sinkhorn, filter_features

import open_clip


def optimal_transport(logits, image_weights, text_weights):
    eps = 0.1
    sim = logits
    sim = sim.permute(2, 0, 1)  # n_cls x M x N

    wdist = 1.0 - sim
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK, image_weights, text_weights)
        T = T.permute(1, 2, 0)
    assert not torch.isnan(T).any()

    return torch.sum(T * logits, dim=(0, 1)).unsqueeze(0)


@torch.no_grad()
def evaluate_synonym(val_loader, model, text_features, mask, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    print("number of test samples: {}".format(len(val_loader)))
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    save_dir = f"./features/pre_extracted_feat/evaclip/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isfile(
            f"./features/pre_extracted_feat/evaclip/seed{args.seed}/{args.test_set}.pth"):
        print("Extracting features for: {}".format(args.test_set))
        all_data = []
        for i, (images, target) in enumerate(tqdm(val_loader, total=len(val_loader))):
            assert isinstance(images, list)
            for k in range(len(images)):
                images[k] = images[k].cuda(non_blocking=True)
            images = torch.cat(images, dim=0)
            target = target.cuda(non_blocking=True)

            with torch.amp.autocast('cuda'):
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = filter_features(image_features, scale=0.8)

            all_data.append((image_features, target))

        save_path = os.path.join(save_dir, f"{args.test_set}.pth")
        torch.save(all_data, save_path)
        print(f"Successfully save image features (bs-{args.batch_size}) to [{save_path}]")

    pre_features = torch.load(os.path.join(save_dir, f"{args.test_set}.pth"), weights_only=False)

    end = time.time()
    for i, (image_features, target) in enumerate(tqdm(pre_features, total=len(pre_features))):
        # optimal transport
        n_views = image_features.size(0)
        n_prompt = mask.size(1)

        output = image_features @ text_features.t()
        output = output.view(n_views, -1, n_prompt).permute(0, 2, 1).contiguous()  # n_view x n_prompt x c

        image_weights, text_weights = get_entropy_weight(output * 100.0, mask)
        image_weights, text_weights = image_weights.to(output.device), text_weights.to(output.device)
        output_ot = optimal_transport(output, image_weights, text_weights)

        acc1, = accuracy(output_ot, target, topk=(1,))
        top1.update(acc1[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    print(f'\n *  {args.test_set}')
    progress.display_summary()


@torch.no_grad()
def main_worker_synonym(args):
    # create model
    model, _, preprocess = open_clip.create_model_and_transforms(
        'EVA02-B-16',
        pretrained='clip_ckpt/eva02_base_patch16_clip_224.merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    )
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')
    model = model.cuda()

    # Image Transformation
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    data_transform = AttnAugmenter(base_transform, preprocess, n_views=args.batch_size)

    # Build dataloader
    val_dataset = build_dataset(args.test_set, data_transform, args.data, mode='test')
    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print("Evaluating: {}".format(args.test_set))
    classnames = get_classnames(args.test_set)

    # get LLM descriptors
    dataset_name = args.test_set
    if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2']:
        description_file = os.path.join(args.descriptor_path, 'imagenet.json')
    else:
        description_file = os.path.join(args.descriptor_path, f'{dataset_name}.json')
    print(f'Using description file: {description_file}')
    llm_descriptions = json.load(open(description_file))

    print('Preparing text features...')
    # get synonyms
    synonym_file = os.path.join(args.descriptor_path.replace('image_datasets', 'synonyms'), f'{dataset_name}.json')
    print(f'Using synonym file: {synonym_file}')
    synonyms = json.load(open(synonym_file))

    # prepare text features
    text_features = []
    template = CUSTOM_TEMPLATES[dataset_name]
    for idx, classname in enumerate(tqdm(classnames, total=len(classnames))):
        prompts = []
        for synonym in synonyms[classname]:
            prompt = template.format(synonym.replace("_", " "))
            prompts.append(prompt + '.')

            # get descriptions
            assert len(llm_descriptions[classname]) >= args.num_descriptor
            for i in range(len(llm_descriptions[classname])):
                description = llm_descriptions[classname][i]
                if dataset_name not in ['sun397', 'imagenet', 'imagenet_a', 'imagenetv2']:
                    description = re.sub(r'{}'.format(re.escape(classname)), synonym, description, flags=re.IGNORECASE)
                prompt_desc = prompt + '. ' + description
                prompts.append(prompt_desc)

        text_inputs = tokenizer(prompts)
        text_inputs = text_inputs.cuda()

        with torch.amp.autocast('cuda'):
            text_outputs = model.encode_text(text_inputs)
            text_features.append(text_outputs / text_outputs.norm(dim=-1, keepdim=True))  # n_desc x d

    # padding text features
    n_cls = len(classnames)
    n_dim = text_features[0].size(1)
    n_desc = [t.size(0) for t in text_features]
    n_prompt = max(n_desc)  # n_prompt=max(n_desc)
    device = text_features[0].device
    padding_text_features = torch.zeros((n_cls, n_prompt, n_dim), device=device)
    for i in range(n_cls):
        padding_text_features[i, :n_desc[i], :] = text_features[i]
    text_features = padding_text_features.view(-1, n_dim).contiguous()  # (n_cls x n_prompt) x d
    mask = torch.arange(n_prompt, device=device)[None, :] < torch.tensor(n_desc, device=device)[:, None]
    del padding_text_features

    # start evaluate
    evaluate_synonym(val_loader, model, text_features, mask, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWT for EVA02-CLIP')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('--resolution', default=224, type=int, help='image resolution')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--descriptor_path', type=str)
    parser.add_argument('--num_descriptor', type=int, default=50)

    args = parser.parse_args()
    set_random_seed(args.seed)
    args.device = 'cuda:0'

    datasets = args.test_set.split("/")

    results = {}
    for set_id in datasets:
        args.test_set = set_id
        main_worker_synonym(args)

        print("======== Result Summary ========")
        print("\t\t [set_id] \t\t Top-1 acc.")
        for id in results.keys():
            print("{}".format(id), end="	")
        print("\n")
        for id in results.keys():
            print("{:.2f}".format(results[id][0]), end="	")
        print("\n")
