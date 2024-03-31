from utils.dataset_utils import *

class ImageDataset(Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing

        self.image_dir = self.get_images_list(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=16)

        prompt = get_text_prompt(
            file_path=train_data,
            text_prompt=self.single_img_prompt,
            fallback_prompt=self.fallback_prompt,
            ext_types=self.img_types,  
            use_caption=True
        )
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        img, prompt, prompt_ids = self.image_batch(index)
        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example