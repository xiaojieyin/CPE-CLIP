import json
from openai import OpenAI

import itertools

import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.getcwd())

from data.cls_to_names import (
    oxford_flowers_classes,
    dtd_classes,
    oxford_pets_classes,
    stanford_cars_classes,
    ucf101_classes,
    caltech101_classes,
    food101_classes,
    sun397_classes,
    fgvc_aircraft_classes,
    eurosat_classes,
    imagenet_classes
)

CLASSES_DICT = {
    "oxford_flowers": oxford_flowers_classes,
    "dtd": dtd_classes,
    "oxford_pets": oxford_pets_classes,
    "stanford_cars": stanford_cars_classes,
    "ucf101": ucf101_classes,
    "caltech101": caltech101_classes,
    "food101": food101_classes,
    "sun397": sun397_classes,
    "fgvc_aircraft": fgvc_aircraft_classes,
    "eurosat": eurosat_classes,
    "imagenet": imagenet_classes,
}

DATASETS_ASSISTANT = {
    "fgvc_aircraft": ' You are an expert in aircraft model recognition.',
    "dtd": '',
    "oxford_flowers": ' You are an expert in flower species recognition.',
    "food101": ' You are an expert in delicious food recognition.',
    "ucf101": ' You are an expert in physical activity recognition.',
    "oxford_pets": ' You are an expert in pet recognition.',
    "eurosat": '',
    "imagenet": '',
    "sun397": '',
    "caltech101": '',
    "stanford_cars": ' You are an expert in car recognition.',
}

LLM_MODEL = {
    'llama': "llama-3.1-405b",
    'gpt': "gpt-3.5-turbo",
    'gpt4': "GPT-4-turbo",
    'claude': "claude-3.5-sonnet",
    'o1': "o1-preview",
    'o1-mini': "o1-mini",
}

DATASETS_CONCEPT = {
    "oxford_pets": " from OxfordPets",
    "oxford_flowers": ", a type of flower",
    "fgvc_aircraft": " from Fine-Grained Visual Classification of Aircraft",
    "dtd": ", which is a type of texture, or a type of pattern",
    "eurosat": "",
    "stanford_cars": "a car",
    "food101": " from Food101",
    "sun397": " a scene",
    "caltech101": "",
    "ucf101": " a person doing",
    "imagenet": " in imagenet dataset"
}


# Generate Prompts.
def generate_prompt(category_name: str, dataset_concept: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: Tell me in ten words or less what are some common ways of referring to a {category_name}{dataset_concept}?
A: Common ways to refer to {category_name} include:
-
-
"""


def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if
            (descriptor != '') and (descriptor.startswith('- '))]


def response_gpt(response):
    if response is None:
        return []
    response = response[0].message.content
    print(response)
    descriptor_list = []
    for res in response.split('\n'):
        res = res.replace('*', '')
        if (res != '') and (res.startswith('- ')):
            descriptor_list.append(res[2:].strip().lower())
        elif (res != '') and (res[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']):
            descriptor_list.append(res[3:].strip().lower())
    return descriptor_list


# generator
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))


def build_synonym(dataset, llm_model):
    class_list = CLASSES_DICT[dataset]
    class_list = [name.replace("_", " ") for name in class_list]
    dataset_concept = DATASETS_CONCEPT[dataset]

    prompts = [generate_prompt(category, dataset_concept) for category in class_list]

    API_KEY = ""
    BASE_URL = ""

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    completion = []
    for prompt in tqdm(prompts, total=len(prompts)):
        completion.append(client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system",
                 "content": f"You are a helpful and honest English assistant.{DATASETS_ASSISTANT[dataset]}"},
                {"role": "user", "content": prompt}],
            temperature=0
        ))

    synonym_list = [response_gpt(response_text.choices) for response_text in completion]
    synonyms = {cat: descr for cat, descr in zip(CLASSES_DICT[dataset], synonym_list)}

    return synonyms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dtd')
    args = parser.parse_args()

    synonyms = build_synonym(args.dataset, LLM_MODEL['claude'].lower())
    print(synonyms)
    save_path = os.path.join('descriptions/synonyms-gpt', f'{args.dataset.lower()}.json')

    with open(save_path, 'w') as outfile:
        outfile.write(json.dumps(synonyms, indent=4))
