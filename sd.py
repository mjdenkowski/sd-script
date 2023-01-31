#!/usr/bin/env python

# License: Public Domain

import argparse
import hashlib
from itertools import product
import time

from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, \
    StableDiffusionPipeline
from PIL.PngImagePlugin import PngInfo
import torch


MODEL_VERSION_2_1 = '2.1'
MODEL_VERSIONS = [MODEL_VERSION_2_1]
MODELS = {MODEL_VERSION_2_1: 'stabilityai/stable-diffusion-2-1'}

SCHEDULER_ID_DPMS = 'dpms'
SCHEDULER_ID_E = 'e'
SCHEDULER_ID_EA = 'ea'
SCHEDULER_IDS = [SCHEDULER_ID_DPMS, SCHEDULER_ID_E, SCHEDULER_ID_EA]
SCHEDULERS = {SCHEDULER_ID_DPMS: DPMSolverMultistepScheduler,
              SCHEDULER_ID_E: EulerDiscreteScheduler,
              SCHEDULER_ID_EA: EulerAncestralDiscreteScheduler}

DEVICE_CPU = 'cpu'
DEVICE_CUDA = 'cuda'
DEVICE_MPS = 'mps'
DEVICES = [DEVICE_CPU, DEVICE_CUDA, DEVICE_MPS]


def generate_images(args: argparse.Namespace):

    # Populate seeds
    seeds = args.seed if args.seed is not None else [int(time.time())]
    if len(seeds) == 1:
        seeds = [seeds[0] + i for i in range(args.num_images)]

    # Load model
    model = MODELS[args.version]
    scheduler_class = SCHEDULERS[args.scheduler]
    scheduler = scheduler_class.from_pretrained(model, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model, scheduler=scheduler)
    pipe = pipe.to(args.device)
    pipe.enable_attention_slicing()

    # Iterate over settings
    for seed, guidance_scale, num_inference_steps in product(seeds, args.guidance_scale, args.num_inference_steps):

        # Generate an image using the specified settings
        image = pipe(prompt=args.prompt,
                     num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale,
                     negative_prompt=args.negative_prompt,
                     generator=torch.Generator().manual_seed(seed)).images[0]

        # Populate generation information
        pnginfo = PngInfo()
        pnginfo.add_text('sd:model', model)
        pnginfo.add_text('sd:prompt', args.prompt)
        if args.negative_prompt:
            pnginfo.add_text('sd:negative_prompt', args.negative_prompt)
        pnginfo.add_text('sd:seed', str(seed))
        pnginfo.add_text('sd:guidance_scale', str(guidance_scale))
        pnginfo.add_text('sd:scheduler', scheduler_class.__name__)
        pnginfo.add_text('sd:num_inference_steps', str(num_inference_steps))

        # Save image
        both_prompts = args.prompt + (args.negative_prompt if args.negative_prompt is not None else '')
        prompts_sha1_short = hashlib.sha1(both_prompts.encode()).hexdigest()[:7]
        fname = f'{args.output_prefix}-{args.version}-{prompts_sha1_short}-{seed}-{guidance_scale}-{args.scheduler}-' \
                f'{num_inference_steps}.png'
        image.save(fname, pnginfo=pnginfo)


def main():

    parser = argparse.ArgumentParser(description='Run Stable Diffusion')
    parser.add_argument('--prompt', '-p',
                        type=str,
                        required=True,
                        help='Prompt')
    parser.add_argument('--output-prefix', '-o',
                        type=str,
                        required=True,
                        help='Images are written to `output_prefix`.`seed`.png')
    parser.add_argument('--num-images',
                        '-n',
                        type=int,
                        default=1,
                        help='Number of images to generate. Default %(default)s.')
    parser.add_argument('--negative-prompt',
                        '-np',
                        type=str,
                        default=None,
                        help='Negative prompt. Default %(default)s.')
    parser.add_argument('--seed',
                        '-s',
                        nargs='+',
                        type=int,
                        default=None,
                        help='If a single value, the starting seed for generating `num_images` images that use '
                             '`seed + 1`, `seed + 2`, etc. If multiple values, the seeds to use (`num_images` '
                             'ignored). Defaults to current time in seconds.')
    parser.add_argument('--num-inference-steps',
                        '-ni',
                        nargs='+',
                        type=int,
                        default=[15],
                        help='Number of inference steps. Can specify multiple values to iterate over. '
                             'Default %(default)s.')
    parser.add_argument('--guidance-scale',
                        '-gs',
                        nargs='+',
                        type=float,
                        default=[7.5],
                        help='Guidance scale. Can specify multiple values to iterate over. Default %(default)s.')
    parser.add_argument('--version',
                        '-v',
                        choices=MODEL_VERSIONS,
                        default=MODEL_VERSION_2_1,
                        help='Model version. Default %(default)s.')
    parser.add_argument('--scheduler',
                        '-sc',
                        choices=SCHEDULER_IDS,
                        default=SCHEDULER_ID_DPMS,
                        help='Model version. Default %(default)s.')
    parser.add_argument('--device',
                        '-d',
                        choices=DEVICES,
                        default=DEVICE_CPU,
                        help='Inference device. Default %(default)s.')

    args = parser.parse_args()
    generate_images(args)


if __name__ == '__main__':
    main()
