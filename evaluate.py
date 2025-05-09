import argparse
import time
import os
import json
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

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
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


def Sinkhorn(K, u, v):
    assert (u >= 0).all(), "Input u must be non-negative."
    assert (v >= 0).all(), "Input v must be non-negative."
    assert torch.isfinite(K).all(), "K contains NaN or Inf."

    r = torch.ones_like(u, device=u.device)
    c = torch.ones_like(v, device=v.device)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T


def optimal_transport(logits, logit_scale, image_weights, text_weights):
    logits = logits.to('cuda:1')
    logit_scale = logit_scale.to('cuda:1')
    image_weights = image_weights.to('cuda:1')
    text_weights = text_weights.to('cuda:1')

    eps = 0.1
    sim = logits / logit_scale.exp()
    sim = sim.permute(2, 0, 1).contiguous()  # n_cls x M x N

    wdist = 1.0 - sim
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK, image_weights, text_weights)
        T = T.permute(1, 2, 0).contiguous()
    assert not torch.isnan(T).any()

    return torch.sum(T * logits, dim=(0, 1)).unsqueeze(0).to('cuda:0')


@torch.no_grad()
def get_entropy_weight(output, mask, img_t=0.5, text_t=0.5):
    device = 'cuda:1'
    output = output.to(device)
    mask = mask.to(device)

    with torch.amp.autocast('cuda'):
        # get weights for images
        image_entropy = calculate_batch_entropy(output.sum(dim=1) / mask.sum(dim=1).float())
        image_weights = F.softmax(-image_entropy / img_t, dim=-1)

        # get weights for descriptors
        _, n_des, n_cls = output.shape
        center = output[0].sum(dim=0) / mask.sum(dim=1).float()
        anchor = center.view(1, 1, n_cls).repeat(n_des, n_cls, 1).contiguous()
        output_des = output[0].unsqueeze(-1)
        del center
        del output
        scatter_indices = torch.arange(n_cls, device=device).view(1, n_cls, 1).repeat(n_des, 1, 1).contiguous()
        anchor.scatter_(dim=2, index=scatter_indices, src=output_des)  # n_des, n_cls, n_cls
        del output_des
        text_entropy = calculate_batch_entropy(anchor)
        text_weights = F.softmax((-text_entropy.t() / text_t).masked_fill(~mask, -float('inf')), dim=-1)  # n_cls, n_des

    return image_weights, text_weights


@torch.no_grad()
def filter_features(features, scale):
    n_views = int(features.size(0) * scale)

    anchor = features[0].unsqueeze(0)  # (1, 512)
    others = features[1:]  # (N, 512)
    sim = (anchor @ others.T).squeeze(0)  # (N,)

    others = others[torch.argsort(sim, descending=True)]
    selected_features = torch.cat([anchor, others[:n_views]], dim=0)

    return selected_features


@torch.no_grad()
def pre_extract_image_feature(val_loader, clip_model, args):
    save_dir = f"./features/pre_extracted_feat/{args.arch.replace('/', '')}/bs{args.batch_size}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    for images, target in tqdm(val_loader):
        assert isinstance(images, list)
        for k in range(len(images)):
            images[k] = images[k].cuda(non_blocking=True)
        images = torch.cat(images, dim=0)
        target = target.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = filter_features(image_features, scale=0.8)

        all_data.append((image_features, target))

    save_path = os.path.join(save_dir, f"{args.test_set}.pth")
    torch.save(all_data, save_path)
    print(f"Successfully save image features to [{save_path}]")


@torch.no_grad()
def AWT_evaluation_synonym(clip_model, args):
    dataset_name = args.test_set
    print("Evaluating: {}".format(dataset_name))
    classnames = get_classnames(dataset_name)

    # get LLM descriptors
    if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2']:
        description_file = os.path.join(args.descriptor_path, 'imagenet.json')
    else:
        description_file = os.path.join(args.descriptor_path, f'{dataset_name}.json')
    print(f'Using description file: {description_file}')
    llm_descriptions = json.load(open(description_file))

    # get synonyms
    if dataset_name.startswith('imagenet'):
        synonym_file = os.path.join(args.descriptor_path.replace('image_datasets', 'synonyms'), 'imagenet.json')
    else:
        synonym_file = os.path.join(args.descriptor_path.replace('image_datasets', 'synonyms'), f'{dataset_name}.json')
    print(f'Using synonym file: {synonym_file}')
    synonyms = json.load(open(synonym_file))

    # ============== prepare text features ==============
    text_features = []
    template = CUSTOM_TEMPLATES[dataset_name]
    for idx, classname in enumerate(classnames):
        prompts = []
        for synonym in synonyms[classname]:
            prompt = template.format(synonym.replace("_", " "))
            prompts.append(prompt + '.')

            # get descriptions
            assert len(llm_descriptions[classname]) >= args.num_descriptor
            for i in range(len(llm_descriptions[classname])):
                description = llm_descriptions[classname][i]
                if dataset_name not in ['sun397', 'imagenet', 'imagenet_a', 'imagenetv2', 'imagenet_r',
                                        'imagenet_sketch', 'UCF101', 'K600']:
                    description = re.sub(r'{}'.format(re.escape(classname)), synonym, description, flags=re.IGNORECASE)
                prompt_desc = prompt + '. ' + description
                prompts.append(prompt_desc)
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(args.device)
        with torch.amp.autocast('cuda'):
            embeddings = clip_model.encode_text(prompts).float()
            text_features.append(embeddings / embeddings.norm(dim=-1, keepdim=True))  # n_desc x d

    # padding text features
    n_cls = len(classnames)
    n_dim = text_features[0].size(1)
    n_desc = [t.size(0) for t in text_features]
    n_prompt = max(n_desc)  # n_prompt=max(n_desc)
    padding_text_features = torch.zeros((n_cls, n_prompt, n_dim), device=args.device)
    for i in range(n_cls):
        padding_text_features[i, :n_desc[i], :] = text_features[i]
    text_features = padding_text_features.view(-1, n_dim).contiguous()  # (n_cls x n_prompt) x d
    mask = torch.arange(n_prompt, device=args.device)[None, :] < torch.tensor(n_desc, device=args.device)[:, None]
    del padding_text_features

    # ==============  calculate logits for each image ==============
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    pre_features = torch.load(
        f"./features/pre_extracted_feat/{args.arch.replace('/', '')}/bs{args.batch_size}/seed{args.seed}/{dataset_name}.pth",
        weights_only=False)

    print("number of test samples: {}".format(len(pre_features)))
    progress = ProgressMeter(len(pre_features), [batch_time, top1], prefix='Test: ')

    end = time.time()
    for i, (image_features, target) in enumerate(pre_features):
        image_features = image_features.to('cuda:0')
        target = target.to('cuda:0')
        n_views = image_features.size(0)

        output = clip_model.logit_scale.exp() * image_features @ text_features.t()
        output = output.view(n_views, -1, n_prompt).permute(0, 2, 1).contiguous()  # n_view x n_prompt x c

        image_temperature = 0.5
        text_temperature = 0.5
        image_weights, text_weights = get_entropy_weight(output, mask, img_t=image_temperature,
                                                          text_t=text_temperature)
        image_weights, text_weights = image_weights.to(args.device), text_weights.to(args.device)
        output_ot = optimal_transport(output, clip_model.logit_scale, image_weights, text_weights)

        acc1, = accuracy(output_ot, target, topk=(1,))
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

    # pre extract image feature
    if not os.path.isfile(
            f"./features/pre_extracted_feat/{args.arch.replace('/', '')}/bs{args.batch_size}/seed{args.seed}/{args.test_set}.pth"):
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        data_transform = AttnAugmenter(base_transform, preprocess, n_views=args.batch_size, resolution=args.resolution)
        print("Extracting features for: {}".format(args.test_set))
        val_dataset = build_dataset(args.test_set, data_transform, args.data, mode='test')
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        pre_extract_image_feature(val_loader, clip_model, args)

    # testing start
    results = AWT_evaluation_synonym(clip_model, args)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWT evaluation')
    parser.add_argument('--test_set', type=str, help='dataset name')
    parser.add_argument('--data', type=str, help='data path')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--descriptor_path', type=str)
    parser.add_argument('--num_descriptor', type=int, default=50)

    parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N')
    parser.add_argument('--resolution', default=224, type=int, help='image resolution')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--num_synonym', type=int, default=-1)

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
