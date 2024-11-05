import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images')
    parser.add_argument('--output-path', type=str, help='path of output images')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.input_path):
        print(f'Cannot find input path: {args.input_path}')
        exit()
    if not os.path.exists(args.output_path):
        print(f'Cannot find output path: {args.output_path}')
        exit()
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find ckpt path: {args.ckpt_path}')
        exit()

    # Define hyperparameters and transformations
    ref_size = 512
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize MODNet and load pre-trained weights
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # Process each image in the input path
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print(f'Processing image: {im_name}')

        # Load and prepare the image
        im = Image.open(os.path.join(args.input_path, im_name))
        im = np.asarray(im)
        if im.shape[2] == 4:
            im = im[:, :, :3]  # Drop alpha if it exists
        im = Image.fromarray(im)
        im = im_transform(im)
        im = im[None, :, :, :]

        # Resize image
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh, im_rw = im_h, im_w
        im_rw, im_rh = im_rw - im_rw % 32, im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # Run MODNet inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        # Convert matte to 0-255 scale and invert if necessary
        matte = (matte * 255).astype("uint8")
        if matte.mean() > 128:  # Invert if the person appears as black
            matte = 255 - matte

        # Save matte as a mask
        matte_name = f"{os.path.splitext(im_name)[0]}_mask.png"
        matte_path = os.path.join(args.output_path, matte_name)
        Image.fromarray(matte, mode="L").save(matte_path)

        # Load original image and apply mask as alpha
        original_image = Image.open(os.path.join(args.input_path, im_name)).convert("RGBA")
        mask = Image.open(matte_path).convert("L")  # Load as grayscale for alpha channel
        original_image.putalpha(mask)

        # Save result with transparent background
        output_path = os.path.join(args.output_path, f"{os.path.splitext(im_name)[0]}_transparent.png")
        original_image.save(output_path)
        original_image.show()
        print(f"Saved transparent image to: {output_path}")
