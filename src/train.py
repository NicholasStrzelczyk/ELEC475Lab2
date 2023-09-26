import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import AdaIN_net as net
from torchvision import transforms


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    image_size = 512
    device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('-content_image', type=str, help='test image')
    parser.add_argument('-style_image', type=str, help='style image')
    parser.add_argument('-encoder_file', type=str, help='encoder weight file')
    parser.add_argument('-decoder_file', type=str, help='decoder weight file')
    parser.add_argument('-alpha', type=float, default=1.0, help='Level of style transfer, value between 0 and 1')
    parser.add_argument('-cuda', type=str, help='[y/N]')

    opt = parser.parse_args()
    content_image = Image.open(opt.content_image)
    style_image = Image.open(opt.style_image)
    output_format = opt.content_image[opt.content_image.find('.'):]
    decoder_file = opt.decoder_file
    encoder_file = opt.encoder_file
    alpha = opt.alpha
    use_cuda = False
    if opt.cuda == 'y' or opt.cuda == 'Y':
        use_cuda = True
    out_dir = './output/'
    os.makedirs(out_dir, exist_ok=True)

    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
    decoder = net.encoder_decoder.decoder
    decoder.load_state_dict(torch.load(decoder_file, map_location='cpu'))
    model = net.AdaIN_net(encoder, decoder)

    model.to(device=device)
    model.eval()

    print('model loaded OK!')

