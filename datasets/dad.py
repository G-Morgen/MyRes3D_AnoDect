import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random

#from utils import load_value_file
def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    #return np.load(path)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, view, image_type, frame_indices, sample_duration, image_loader):
    video = []
    v = view
    for i in frame_indices:
        image_path_IR = os.path.join(video_dir_path, v, "IR",'{:02d}.png'.format(i))
        image_path_D = os.path.join(video_dir_path, v ,"depth",'{:02d}.png'.format(i))
        if image_type == 'IR':
            if os.path.exists(image_path_IR):
                video.append(image_loader(image_path_IR))

        elif image_type == 'depth':
            if os.path.exists(image_path_D):
                video.append(image_loader(image_path_D))

        elif image_type == 'D+IR': 
            if os.path.exists(image_path_D) and os.path.exists(image_path_IR):
                video.append(image_loader(image_path_D))
                video.append(image_loader(image_path_IR))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset, motion):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            if motion == "all":
                video_names.append(key)
                annotations.append(value['annotations'])
            elif label == motion:
                video_names.append(key)
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, motion, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset, motion)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            #print(video_path)
            continue

        #n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = 45#int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 0
        end_t = 44
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': video_names[i]
        }
        #print (sample["video"])
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, 1+n_frames))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class Dad(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 motion,
                 view,
                 image_type,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        #print (root_path,annotation_path,subset,n_samples_for_each_video,sample_duration)
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, motion, n_samples_for_each_video,
            sample_duration)
    
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.sample_duration = sample_duration
        self.view = view
        self.image_type = image_type
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices'] #[1...45]
        #print (frame_indices)
        if self.temporal_transform is not None:
           frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, self. view, self.image_type, frame_indices, self.sample_duration)
        #clip = [torch.from_numpy(i) for i in clip]
        #print (clip[0].size())
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        #print (clip[0].size())
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        #clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        #print (clip.size())
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    #jester = Jester("~/Jester/20bn-jester-v1/","annotation_anomaly/jester.json",
    #'training', spatial_transform=None, temporal_transform=None, target_transform=None, sample_duration=16)
    a,b = make_dataset("/usr/home/sut/N_AD","/usr/home/sut/3D-ResNets-PyTorch/annotation_tumad/jester.json","validation",
    1,32)
    print (1)
