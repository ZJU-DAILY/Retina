import math
from typing import ClassVar, List, Optional, Tuple, Union
import random
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor

from LLM4IR.utils.processing_utils import BaseVisualRetrieverProcessor


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


class MonoQwen2Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Processor for ColQwen2.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_prefix: ClassVar[str] = "Query: "
    candidate_prefix: ClassVar[str] = "Candidate: "
    instruction_prefix: ClassVar[str] = "Instruction: "
    augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"
    Yes_token_id: ClassVar[str] = 9454
    No_token_id: ClassVar[str] = 2753
    
    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __init__(self, *args, **kwargs):
        num_image_tokens = kwargs.pop("num_image_tokens", 768)
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"
        self.im_end_token_id = 151645
        self.im_start_token_id = 151644
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = num_image_tokens * 28 * 28
        self.factor = 28
        self.max_ratio = 200

    @staticmethod
    def smart_resize_helper(
        width: int,
        height: int,
        factor: int,
        max_ratio: int,
        min_pixels: int,
        max_pixels: int,
    ) -> Tuple[int, int]:
        """
        Returns the image size so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, "
                f"got {max(height, width) / min(height, width)}"
            )

        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        return h_bar, w_bar
    @staticmethod
    def format_prompt(
        text: str,
        prompt: str = ''
    ) -> str:
        if text.strip() == '<image>':
            text = text + 'Describe the image.'
            
        formatted_text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        
        formatted_text = f"{prompt}{formatted_text}"
        
        return formatted_text

    @staticmethod
    def flatten_and_length(images: List[List[Image.Image]]) -> Tuple[List[Image.Image], List[int]]:
        flattened_images = []
        lengths = []
        for sublist in images:
            # Filter out None values and count the remaining valid images
            valid_images = [image for image in sublist if image is not None]
            flattened_images.extend(valid_images)
            lengths.append(len(valid_images))
        return flattened_images, lengths
    
    def smart_resize(self, image: Image.Image) -> Image.Image:
        """
        Resize and convert the image to the required format.
        """
        image_size = image.size
        resized_height, resized_width = self.smart_resize_helper(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            max_ratio=self.max_ratio,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return image.convert("RGB").resize((resized_width, resized_height))
    def process(
        self,
        input: list,
    ) -> BatchFeature:
        """
        Process images for ColQwen2.
        """
        
        texts, images = zip(*[(item['txt'], item['img']) for item in input if 'txt' in item and 'img' in item])
        
        prompt_list = [item['prompt'] for item in input if 'prompt' in item]
        
        images, images_sublist_length = self.flatten_and_length(images)
        
        resized_images: List[Image.Image] = [self.smart_resize(image) for image in images] if images else None

        texts = [self.format_prompt(texts[i], prompt_list[i]) for i in range(len(texts))]
        
        batch_doc = self(
            text=texts,
            images=resized_images,
            padding="longest",
            return_tensors="pt",
        )
        
        # NOTE: The following code is a hack to make sure the scatter in DDP is done correctly when training
        # on multiple GPUs.
        if "pixel_values" in batch_doc:
            offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]

            # separate pixel_values for each image
            pixel_values = torch.split(batch_doc["pixel_values"], offsets.tolist())
            pixel_values_max_length = max([len(pv) for pv in pixel_values])

            pixel_values = [
                torch.cat([pv, torch.zeros((pixel_values_max_length - len(pv), pv.shape[1]), dtype=pv.dtype, device=pv.device)])
                for pv in pixel_values
            ]
            batch_doc['images_sublist_length'] = torch.tensor(images_sublist_length)
            
            image_grid_thw = torch.split(batch_doc["image_grid_thw"], images_sublist_length)
            pixel_values = torch.split(torch.stack(pixel_values), images_sublist_length)
            
            images_sublist_max_length = max(images_sublist_length)
            
            image_grid_thw = [
                torch.cat([img_grid_thw, torch.zeros((images_sublist_max_length - len(img_grid_thw), img_grid_thw.shape[1]), dtype=img_grid_thw.dtype, device=img_grid_thw.device)])
                if len(img_grid_thw) < images_sublist_max_length else img_grid_thw
                for img_grid_thw in image_grid_thw
            ]
            pixel_values = [
                torch.cat([pv, torch.zeros((images_sublist_max_length - len(pv), pv.shape[1], pv.shape[2]), dtype=pv.dtype, device=pv.device)])
                if len(pv) < images_sublist_max_length else pv
                for pv in pixel_values
            ]
            batch_doc["image_grid_thw"] = torch.stack(image_grid_thw)
            batch_doc["pixel_values"] = torch.stack(pixel_values)

        return batch_doc
    
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_sparse_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        height_new, width_new = self.smart_resize_helper(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            max_ratio=self.max_ratio,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
