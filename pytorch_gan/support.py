import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from BasicDiscriminator import BasicDiscriminator
from BasicGenerator import BasicGenerator
from torch.autograd import Variable
from random import randint
import shutil


def train_D(D, G, opt, data, args, train_fake = True):
    cuda = torch.cuda.is_available()
    noise_dim = args.noise_dim
    dual = args.dual_discrim
    batch_size = args.batch_size

    # D Real Training
    D.zero_grad()

    real_image = Variable(data['image'])
    
    binary_label = Variable(torch.ones(real_image.size(0)))
    if dual: class_label = Variable(data['label'])

    if cuda:
        real_image = real_image.cuda()
        binary_label = binary_label.cuda()
        if dual: class_label = class_label.cuda()

    if dual:
        binary_output, class_output = D(real_image)
        binary_output = binary_output.view(binary_output.numel())
        class_output = class_output.view(real_image.size(0), 10, 1, 1)

        D_real_binary_loss = F.binary_cross_entropy(binary_output, binary_label)
        D_real_class_loss = F.multilabel_soft_margin_loss(class_output, class_label)

        D_real_loss = D_real_binary_loss + D_real_class_loss

        ground_truth = torch.max(class_label, 1)[1]
        predicted = (torch.max(class_output, 1)[1]).squeeze()
        correct = (ground_truth == predicted).sum(0)
        acc = float(correct) / float(ground_truth.size(0))
    else:
        output = D(real_image)
        output = output.view(output.numel())
        D_real_loss = F.binary_cross_entropy(output, binary_label)
        
    if not train_fake:
        D_real_loss.backward()
        opt.step()
        if dual:
            return (D_real_loss, acc)
        else:
            return D_real_loss

    # D Fake Training
    D.zero_grad()

    noise = torch.FloatTensor(batch_size, noise_dim, 1, 1).normal_(0, 1)    
    if dual:
        class_label = gen_class_label(batch_size)
        noise = torch.cat([noise, class_label], 1)
        class_label = Variable(class_label)
    noise = Variable(noise)
    binary_label = Variable(torch.zeros(noise.size(0)))

    if cuda:
        noise = noise.cuda()
        binary_label = binary_label.cuda()
        if dual: class_label = class_label.cuda()

    fake_image = G(noise)

    if dual:
        binary_output, class_output = D(fake_image)
        binary_output = binary_output.view(binary_output.numel())
        class_output = class_output.view(batch_size, 10, 1, 1)

        D_fake_binary_loss = F.binary_cross_entropy(binary_output, binary_label)
        D_fake_class_loss = F.multilabel_soft_margin_loss(class_output, class_label)

        D_fake_loss = D_fake_binary_loss + D_fake_class_loss
    else:
        output = D(fake_image)
        output = output.view(output.numel())
        D_fake_loss = F.binary_cross_entropy(output, binary_label)


    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    # Update D
    opt.step()
    return (D_real_loss, D_fake_loss)

def train_G(D, G, opt, args):
    cuda = torch.cuda.is_available()
    noise_dim = args.noise_dim
    dual = args.dual_discrim
    batch_size = args.batch_size

    G.zero_grad()
    D.zero_grad()

    noise = torch.FloatTensor(batch_size, noise_dim, 1, 1).normal_(0, 1)
    # USING REAL LABEL TO UPDATE G
    if dual:
        class_label = gen_class_label(batch_size)
        noise = torch.cat([noise, class_label], 1)
        class_label = Variable(class_label)
    noise = Variable(noise)
    binary_label = Variable(torch.ones(noise.size(0)))

    if torch.cuda.is_available():
        noise = noise.cuda()
        binary_label = binary_label.cuda()
        if dual: class_label = class_label.cuda()
            
    fake_image = G(noise)

    if dual:
        binary_output, class_output = D(fake_image)
        binary_output = binary_output.view(binary_output.numel())
        class_output = class_output.view(batch_size, 10, 1, 1)

        G_binary_loss = F.binary_cross_entropy(binary_output, binary_label)
        G_class_loss = F.multilabel_soft_margin_loss(class_output, class_label)
        G_loss = G_binary_loss + G_class_loss
    else:
        output = D(fake_image)
        output = output.view(output.numel())
        G_loss = F.binary_cross_entropy(output, binary_label)

    G_loss.backward()
    opt.step()
    return G_loss


def gen_class_label(batch_size, classes = 10):
    c = [randint(0,classes - 1) for _ in xrange(batch_size)]
    class_label = torch.zeros(batch_size, classes, 1, 1)
    for row, idx in enumerate(c):                
        class_label[row][idx] = 1
    return torch.FloatTensor(class_label)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './post_refactor/model_best.pth.tar')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)