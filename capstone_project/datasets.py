import numpy as np
import pandas as pd
from scipy.misc import imresize
import logging

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FixedMovingMNISTDataset(Dataset):
    def __init__(self, data, time_buckets, num_frames_in_stack=2, size=300000, transforms=None):
        self.data = data
        self.size = size
        self.num_frames_in_stack = num_frames_in_stack
        self.time_buckets_dict = self._get_time_buckets_dict(time_buckets)
        self._check_data()
        self.candidates_dict = self._get_candidates_differences_dict()
        self.transforms = transforms

    def __getitem__(self, index):
        video_idx = np.random.choice(len(self.data))
        y = np.random.choice(list(self.time_buckets_dict.keys()))

        (x1, x2), difference, (frame1, frame2) = self._get_sample_at_difference(video_idx, y)

        if self.transforms:
            x1 = self.transforms(x1)
            x2 = self.transforms(x2)
            # x1 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x1], dim=0).squeeze(1)
            # x2 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x2], dim=0).squeeze(1)

        y = torch.from_numpy(np.array(y))

        return x1.float(), x2.float(), y.long(), difference, (frame1, frame2)

    def __len__(self):
        return self.size

    def _check_data(self):
        sequence_length = self.data.shape[1]
        max_frame_diff = np.hstack(self.time_buckets_dict.values()).max()
        assert max_frame_diff <= sequence_length-self.num_frames_in_stack, \
            'Cannot have difference of {} when sequence length is {} and number of \
            stacked frames are {}'.format(max_frame_diff, sequence_length, self.num_frames_in_stack)

    def _get_time_buckets_dict(self, time_buckets):
        '''
        Returns a dict, with the bucket idx target
        class (0-indexed) as its key and the time ranges
        for it as its value
        '''
        buckets_dict = dict(zip(range(len(time_buckets)), time_buckets))
        return buckets_dict

    def _get_candidates_differences_dict(self):
        '''
        Returns a dict with the key as the time difference between the frames
        and the value as a list of tuples (start_frame, end_frame) containing
        all the pair of frames with that time difference
        '''
        logging.info('Getting frame differences dictionary...')
        sequence_length = self.data.shape[1]
        max_frame_diff = np.hstack(self.time_buckets_dict.values()).max()

        differences_dict = {}
        differences = range(max_frame_diff+1)
        for diff in differences:
            start_frame = self.num_frames_in_stack-1
            end_frame = start_frame+diff
            while end_frame <= sequence_length-1:
                differences_dict.setdefault(diff, []).append(tuple((start_frame, end_frame)))
                start_frame += 1
                end_frame += 1
        logging.info('Done.')
        return differences_dict

    def _get_sample_at_difference(self, video_idx, bucket_idx):
        '''
        Sampling a time difference from the associated bucket idx,
        sampling a video pair at that difference, and finally returning
        the (stacked) image pairs (tuple), their time difference, and
        the last frame numbers for each pair (tuple)
        '''
        video = self.data[video_idx]
        difference = np.random.choice(self.time_buckets_dict[bucket_idx])
        candidates = self.candidates_dict[difference]
        pair_idx = np.random.choice(len(candidates))
        image1_last_frame, image2_last_frame = candidates[pair_idx]

        image1_frames = range(image1_last_frame-self.num_frames_in_stack+1, image1_last_frame+1)
        image2_frames = range(image2_last_frame-self.num_frames_in_stack+1, image2_last_frame+1)
        image_pair = np.array([video[image1_frames], video[image2_frames]])

        image_pair_idxs = np.array([image1_last_frame, image2_last_frame])

        return torch.from_numpy(image_pair), torch.from_numpy(np.array(difference)), \
                torch.from_numpy(image_pair_idxs)


class RandomMovingMNISTVideoGenerator(object):
    '''Data Handler that creates Moving MNIST videos dataset on the fly'''

    def __init__(self, data, seq_len=200, output_image_size=64, num_digits=1):
        self.data = data
        self.seq_len = seq_len
        self.image_size = output_image_size
        self.output_image_size = output_image_size
        self.num_digits = num_digits
        self.step_length = 0.1

        self.digit_size = data.shape[-1]
        self.frame_size = self.image_size ** 2

        self.indices = np.arange(self.data.shape[0])
        self.row = 0
        np.random.shuffle(self.indices)

    def _get_dims(self):
        return self.frame_size

    def _get_seq_len(self):
        return self.seq_len

    def _get_random_trajectory(self, num_digits):
        canvas_size = self.image_size - self.digit_size

        # Initial position uniform random inside the box
        y = np.random.rand(num_digits)
        x = np.random.rand(num_digits)

        # Choose a random velocity
        theta = np.random.rand(num_digits) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((self.seq_len, num_digits))
        start_x = np.zeros((self.seq_len, num_digits))
        for i in range(self.seq_len):
            # Take a step along velocity
            y += v_y * self.step_length
            x += v_x * self.step_length

            # Bounce off edges
            for j in range(num_digits):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def _overlap(self, a, b):
        '''Put b on top of a'''
        return np.maximum(a, b)

    def _resize(self, data, size):
        output_data = np.zeros((self.seq_len, size, size), dtype=np.float32)
        for i, frame in enumerate(data):
            output_data[i] = imresize(frame, (size, size))
        return output_data

    def __getitem__(self):
        start_y, start_x = self._get_random_trajectory(self.num_digits)

        # Mini-batch data
        data = np.zeros((self.seq_len, self.image_size, self.image_size), dtype=np.float32)

        for n in range(self.num_digits):

            # Get random digit from dataset
            ind = self.indices[self.row]
            self.row += 1
            if self.row == self.data.shape[0]:
                self.row = 0
                np.random.shuffle(self.indices)
            digit_image = self.data[ind, :, :]

            # Generate video
            for i in range(self.seq_len):
                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size
                right = left + self.digit_size
                data[i, top:bottom, left:right] = self._overlap(
                    data[i, top:bottom, left:right], digit_image)

        if self.output_image_size == self.image_size:
            return data
        return self._resize(data, self.output_image_size)


class RandomMovingMNISTDataset(Dataset):
    def __init__(self, data_handler, time_buckets, num_frames_in_stack=2, size=300000):
        self.data_handler = data_handler
        self.size = size
        self.num_frames_in_stack = num_frames_in_stack
        self.time_buckets_dict = self._get_time_buckets_dict(time_buckets)
        self._check_data()
        self.candidates_dict = self._get_candidates_differences_dict()
        self.transforms = transforms

    def __getitem__(self, index):
        video = self.data_handler.__getitem__()
        y = np.random.choice(list(self.time_buckets_dict.keys()))

        (x1, x2), difference, (frame1, frame2) = self._get_sample_at_difference(video, y)

        y = torch.from_numpy(np.array(y))

        return x1.float(), x2.float(), y.long(), difference, (frame1, frame2)

    def __len__(self):
        return self.size

    def _check_data(self):
        sequence_length = self.data_handler._get_seq_len()
        max_frame_diff = np.hstack(self.time_buckets_dict.values()).max()
        assert max_frame_diff <= sequence_length-self.num_frames_in_stack, \
            'Cannot have difference of {} when sequence length is {} and number of \
            stacked frames are {}'.format(max_frame_diff, sequence_length, self.num_frames_in_stack)

    def _get_time_buckets_dict(self, time_buckets):
        '''
        Returns a dict, with the bucket idx target
        class (0-indexed) as its key and the time ranges
        for it as its value
        '''
        buckets_dict = dict(zip(range(len(time_buckets)), time_buckets))
        return buckets_dict

    def _get_candidates_differences_dict(self):
        '''
        Returns a dict with the key as the time difference between the frames
        and the value as a list of tuples (start_frame, end_frame) containing
        all the pair of frames with that time difference
        '''
        logging.info('Getting frame differences dictionary...')
        sequence_length = self.data_handler._get_seq_len()
        max_frame_diff = np.hstack(self.time_buckets_dict.values()).max()

        differences_dict = {}
        differences = range(max_frame_diff+1)
        for diff in differences:
            start_frame = self.num_frames_in_stack-1
            end_frame = start_frame+diff
            while end_frame <= sequence_length-1:
                differences_dict.setdefault(diff, []).append(tuple((start_frame, end_frame)))
                start_frame += 1
                end_frame += 1
        logging.info('Done.')
        return differences_dict

    def _get_sample_at_difference(self, video, bucket_idx):
        '''
        Sampling a time difference from the associated bucket idx,
        sampling a video pair at that difference, and finally returning
        the (stacked) image pairs (tuple), their time difference, and
        the last frame numbers for each pair (tuple)
        '''
        difference = np.random.choice(self.time_buckets_dict[bucket_idx])
        candidates = self.candidates_dict[difference]
        pair_idx = np.random.choice(len(candidates))
        image1_last_frame, image2_last_frame = candidates[pair_idx]

        image1_frames = range(image1_last_frame-self.num_frames_in_stack+1, image1_last_frame+1)
        image2_frames = range(image2_last_frame-self.num_frames_in_stack+1, image2_last_frame+1)
        image_pair = np.array([video[image1_frames], video[image2_frames]])

        image_pair_idxs = np.array([image1_last_frame, image2_last_frame])

        return torch.from_numpy(image_pair), torch.from_numpy(np.array(difference)), \
                torch.from_numpy(image_pair_idxs)


class MazeDataset(Dataset):
    def __init__(self, data, time_buckets, num_channels=3, size=300000, return_embedding=False):
        self.data = data
        self.size = size
        self.num_channels = num_channels
        self.time_buckets_dict = self._get_time_buckets_dict(time_buckets)
        self._check_data()
        self.candidates_differences_dict = self._get_candidates_differences_dict()
        self.transforms = transforms
        self.return_embedding = return_embedding

    def __getitem__(self, index):
        video_idx = np.random.choice(len(self.data))
        y = np.random.choice(list(self.time_buckets_dict.keys()))

        (x1, x2), difference, (frame1, frame2) = self._get_sample_at_difference(video_idx, y)
        y = torch.from_numpy(np.array(y))

        if self.return_embedding:
            x1 = x1.clamp(0, 10)
            x2 = x2.clamp(0, 10)
            return x1.long(), x2.long(), y.long(), difference, (frame1, frame2)
        else:
            x1 = x1.clamp(0, 10) / 10
            x2 = x2.clamp(0, 10) / 10
            return x1.float(), x2.float(), y.long(), difference, (frame1, frame2)

    def __len__(self):
        return self.size

    def _check_data(self):
        max_frame_diff = np.hstack(self.time_buckets_dict.values()).max()
        video = self.data[0]
        num_channels = self.data[0].shape[1]
        assert self.num_channels == num_channels, "Number of channels passed \
            (={}) don't match that in data (={})".format(self.num_channels, num_channels)

        min_seq_len = np.min([maze.shape[0] for maze in self.data])
        assert max_frame_diff <= min_seq_len, \
            'Min sequence length (={}) not long enough for max_frame_diff (={})'\
            .format(min_seq_len, max_frame_diff)

    def _get_time_buckets_dict(self, time_buckets):
        '''
        Returns a dict, with the bucket idx target
        class (0-indexed) as its key and the time ranges
        for it as its value
        '''
        buckets_dict = dict(zip(range(len(time_buckets)), time_buckets))
        return buckets_dict

    def _get_candidates_differences_dict(self):
        '''
        Returns a dict with:
          - key as the sequence length of a video
          - value as a dict with:
            - key as the time difference between the frames
            - value as a list of tuples (start_frame, end_frame) containing
              all the pair of frames with that time difference for a given video
        '''
        logging.info('Getting candidates differences dict for all sequence lengths...')
        differences_dict = {}
        max_frame_diff = np.hstack(self.time_buckets_dict.values()).max()
        for video in self.data:
            seq_len = video.shape[0]
            if not differences_dict.get(seq_len):
                differences_dict[seq_len] = {}
                differences = range(max_frame_diff+1)
                for diff in differences:
                    start_frame, end_frame = 0, diff
                    while end_frame <= seq_len-1:
                        differences_dict[seq_len].setdefault(diff, []).append(tuple((start_frame, end_frame)))
                        start_frame += 1
                        end_frame += 1
        logging.info('Done.')
        return differences_dict

    def _get_sample_at_difference(self, video_idx, bucket_idx):
        '''
        Sampling a time difference from the associated bucket idx for a,
        given video, and returning the image pairs (tuple), their time
        difference, and the last frame numbers for each pair (tuple)
        '''
        video = self.data[video_idx]
        seq_len = video.shape[0]
        difference = np.random.choice(self.time_buckets_dict[bucket_idx])
        candidates = self.candidates_differences_dict[seq_len][difference]
        pair_idx = np.random.choice(len(candidates))
        image1_frame_idx, image2_frame_idx = candidates[pair_idx]
        image_pair_idxs = np.array([image1_frame_idx, image2_frame_idx])
        image_pair = np.array([video[image1_frame_idx], video[image2_frame_idx]])

        return torch.from_numpy(image_pair), torch.from_numpy(np.array(difference)), \
                torch.from_numpy(image_pair_idxs)


class MazeEmbeddingsDataset(Dataset):
    def __init__(self, data, num_channels=3, return_embedding=False):
        self.data = data
        self.num_channels = num_channels
        self._check_data()
        self.return_embedding = return_embedding

    def __getitem__(self, index):
        video = torch.from_numpy(np.array(self.data[index]))

        if self.return_embedding:
            video = torch.stack([img.clamp(0, 10).long() for img in video], dim=0)
        else:
            video = torch.stack([(img.clamp(0, 10)/10).float() for img in video], dim=0)
        return video

    def __len__(self):
        return len(self.data)

    def _check_data(self):
        video = self.data[0]
        num_channels = self.data[0].shape[1]
        assert self.num_channels == num_channels, "Number of channels passed \
            (={}) don't match that in data (={})".format(self.num_channels, num_channels)


class OfflineMovingMNISTDataset(Dataset):
    def __init__(self, X, y, differences, frame_numbers, transforms=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.differences = torch.from_numpy(differences)
        self.frame_numbers = torch.from_numpy(frame_numbers)
        self.transforms = transforms

    def __getitem__(self, index):
        x1, x2 = self.X[index]
        y = self.y[index]
        difference = self.differences[index]
        frame1, frame2 = self.frame_numbers[index]

        if self.transforms:
            x1 = self.transforms(x1)
            x2 = self.transforms(x2)

        return x1.float(), x2.float(), y.long(), difference, (frame1, frame2)

    def __len__(self):
        return len(self.y)
