import argparse
import time
import torch
import AdaIN_net as net
import custom_dataset as cds
import torch.optim as opt
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms


def get_train_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def get_batch_factor(num_batches):
    factor = 1
    if num_batches <= 5:
        factor = 1
    elif num_batches <= 20:
        factor = 5
    elif num_batches <= 50:
        factor = 10
    elif num_batches <= 100:
        factor = 25
    elif num_batches <= 200:
        factor = 50
    elif num_batches <= 500:
        factor = 100
    elif num_batches <= 1000:
        factor = 250
    return factor


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count):
    lr = args.learn / (1.0 + args.gamma * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    device = torch.device('cpu')

    # set up argument parser for command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-content_dir', type=str, help='test directory')
    parser.add_argument('-style_dir', type=str, help='style directory')
    parser.add_argument('-learn', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=5e-5, help='Gamma value')
    parser.add_argument('-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-l', type=str, help='Encoder file path')
    parser.add_argument('-s', type=str, help='Decoder file path')
    parser.add_argument('-p', type=str, help='Result image file path')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # get args from the arg parser
    args = parser.parse_args()
    learn = float(args.learn)
    gamma = float(args.gamma)
    n_epochs = int(args.e)
    batch_size = int(args.b)

    content_tf = train_transform()
    content_dataset = cds.custom_dataset(str(args.content_dir), train_transform())
    content_dataloader = DataLoader(content_dataset, batch_size=batch_size, shuffle=True)
    style_tf = train_transform()
    style_dataset = cds.custom_dataset(str(args.style_dir), train_transform())
    style_dataloader = DataLoader(style_dataset, batch_size=batch_size, shuffle=True)

    encoder_path = Path(args.l)
    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    decoder_path = Path(args.s)
    decoder = net.encoder_decoder.decoder

    result_path = Path(args.p)
    use_cuda = str(args.cuda).lower()

    data_length = len(content_dataset)
    n_batches = len(content_dataloader)
    batch_factor = get_batch_factor(n_batches)

    # initialize model and training parameters
    model = net.AdaIN_net(encoder, decoder)
    model.train()
    model.to(device=torch.device('cpu'))
    optimizer = opt.Adam(net.encoder_decoder.decoder.parameters(), lr=learn)
    print('model loaded OK!')

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # begin training the model
    loss_c_train = []
    loss_s_train = []
    loss_train = []
    print("{} training...".format(datetime.now()))
    start_time = time.time()

    for epoch in (range(n_epochs)):
        print("Epoch {}/{}".format(epoch+1, n_epochs))
        loss_c = 0.0
        loss_s = 0.0

        for batch in (range(n_batches)):
            content_images = next(iter(content_dataloader)).to(device)
            style_images = next(iter(style_dataloader)).to(device)
            lc, ls = model(content_images, style_images)
            loss = lc + ls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_c += lc.item()
            loss_s += ls.item()
            if (batch+1) % batch_factor == 0:
                print("completed {}/{} batches".format(batch+1, n_batches))

        adjust_learning_rate(optimizer=optimizer, iteration_count=epoch+1)

        loss_c_train.append(loss_c / data_length)
        loss_s_train.append(loss_s / data_length)
        loss_train.append((loss_c + loss_s) / data_length)

        print("{} Epoch {}, c_loss {}, s_loss {}, total_loss {}".format(
            datetime.now(), epoch+1, loss_c / data_length, loss_s / data_length, (loss_c + loss_s) / data_length))

    end_time = time.time()

    # print final training statistics
    print("Total training time: {}".format(get_train_time(start_time, end_time)))
    print("Final loss value: {}".format(loss_train[-1]))

    # save the decoder and plot file
    torch.save(model.decoder.state_dict(), decoder_path)
    plt.figure(figsize=(12, 7))
    plt.clf()
    plt.plot(loss_train, label='content+style')
    plt.plot(loss_c_train, label='content')
    plt.plot(loss_s_train, label='style')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc=1)
    plt.savefig(result_path)
    plt.show()
