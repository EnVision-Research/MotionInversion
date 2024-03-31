from utils.dataset_utils import *

# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoJsonDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 1,
            frame_step: int = 1,
            json_path: str ="",
            json_data = None,
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key
        self.train_data = self.load_from_json(json_path, json_data)

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.frame_step = frame_step

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        clip_path = nested_data['clip_path'] if 'clip_path' in nested_data else None
        
        extended_data.append({
            self.vid_data_key: data[self.vid_data_key],
            'frame_index': nested_data['frame_index'],
            'prompt': nested_data['prompt'],
            'clip_path': clip_path
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                print(f"Loading JSON from {path}")
                json_data = json.load(jpath)

                return self.build_json(json_data)

        except:
            self.train_data = []
            print("Non-existant JSON path. Skipping.")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_range(self, vr):
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            self.frame_step, 
            self.n_sample_frames
        )
    
    def get_vid_idx(self, vr, vid_data=None):
        frames = self.n_sample_frames

        if vid_data is not None:
            idx = vid_data['frame_index']
        else:
            idx = self.sample_start_idx

        return idx

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        # width, height = self.width, self.height
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        frame_range = self.get_frame_range(vr)
        frames = vr.get_batch(frame_range)
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def train_data_batch(self, index):

        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index] and \
            self.train_data[index]['clip_path'] is not None:

            vid_data = self.train_data[index]

            clip_path = vid_data['clip_path']
            
            # Get video prompt
            prompt = vid_data['prompt']

            video, _ = self.process_video_wrapper(clip_path)

            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids

         # Assign train data
        train_data = self.train_data[index]
        
        # Get the frame of the current index.
        self.sample_start_idx = train_data['frame_index']
        
        # Initialize resize
        resize = None

        video, vr = self.process_video_wrapper(train_data[self.vid_data_key])

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.train_data is not None:
            video, prompt, prompt_ids = self.train_data_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example