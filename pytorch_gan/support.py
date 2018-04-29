import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import utils
from BasicDiscriminator import BasicDiscriminator
from BasicGenerator import BasicGenerator
from torch.autograd import Variable
from random import randint
import shutil


def train_D(D, G, opt, data, args, train_fake = True):
    cuda = torch.cuda.is_available()
    noise_dim = args.noise_dim
    dual = args.dual_discrim
    class_out = args.class_output
    batch_size = args.batch_size

    # D Real Training
    D.zero_grad()

    real_image = Variable(data['image'])
    
    binary_label = Variable(torch.ones(real_image.size(0)))
    if dual or class_out:
        class_label = data['label']
            
        if class_out:
            class_label = class_label.view(real_image.size(0), 11, 1, 1)
            class_idx = Variable(data['idx'])
            
        class_label = Variable(class_label)

    if cuda:
        real_image = real_image.cuda()
        binary_label = binary_label.cuda()
        if dual or class_out:
            class_label = class_label.cuda()
            class_idx = class_idx.cuda()

    if dual:
        # binary_output, class_output = D(real_image)
        # binary_output = binary_output.view(binary_output.numel())
        # class_output = class_output.view(real_image.size(0), 10, 1, 1)

        # D_real_binary_loss = F.binary_cross_entropy(binary_output, binary_label)
        # D_real_class_loss = F.multilabel_soft_margin_loss(class_output, class_label)

        # D_real_loss = D_real_binary_loss + D_real_class_loss
        output = D(real_image, class_label)
        output = output.view(output.numel())
        D_real_loss = F.binary_cross_entropy(output, binary_label)
    elif class_out:
        output = D(real_image)
        output = output.view(real_image.size(0), 11)
        D_real_loss = F.cross_entropy(output, class_idx)
    else:
        output = D(real_image)
        output = output.view(output.numel())
        D_real_loss = F.binary_cross_entropy(output, binary_label)
        
    if not train_fake:
        D_real_loss.backward()
        opt.step()
        if dual:
            return (D_real_binary_loss, D_real_class_loss)
        else:
            return D_real_loss

    # D Fake Training
    D.zero_grad()

    noise, class_label, class_idxs = gen_G_input(args, class_label = dual or class_out, fake_image = class_out)
    noise = Variable(noise)
    if dual or class_out: class_label = Variable(class_label)
    if class_out: class_idxs = Variable(class_idxs)
    

    binary_label = Variable(torch.zeros(noise.size(0)))

    if cuda:
        noise = noise.cuda()
        binary_label = binary_label.cuda()
        if dual or class_out: class_label = class_label.cuda()
        if class_out: class_idxs = class_idxs.cuda()
        

    fake_image = G(noise, class_label)

    if dual:
        # binary_output, class_output = D(fake_image)
        # binary_output = binary_output.view(binary_output.numel())
        # class_output = class_output.view(batch_size, 10, 1, 1)

        # D_fake_binary_loss = F.binary_cross_entropy(binary_output, binary_label)
        # D_fake_class_loss = F.multilabel_soft_margin_loss(class_output, class_label)

        # D_fake_loss = D_fake_binary_loss + D_fake_class_loss

        output = D(fake_image, class_label)
        output = output.view(output.numel())
        D_fake_loss = F.binary_cross_entropy(output, binary_label)
    elif class_out:
        output = D(fake_image)
        output = output.view(noise.size(0), 11)
        D_fake_loss = F.cross_entropy(output, class_idxs)
    else:
        output = D(fake_image)
        output = output.view(output.numel())
        D_fake_loss = F.binary_cross_entropy(output, binary_label)


    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    # Update D
    opt.step()
    if dual:
        return (D_real_loss, D_real_loss, D_fake_loss, D_fake_loss)
    else:
        return (D_real_loss, D_fake_loss)

def train_G(D, G, opt, args):
    cuda = torch.cuda.is_available()
    noise_dim = args.noise_dim
    dual = args.dual_discrim
    class_out = args.class_output
    batch_size = args.batch_size

    G.zero_grad()
    D.zero_grad()

    noise, class_label, class_idxs = gen_G_input(args, dual or class_out)
    noise = Variable(noise)

    if dual or class_out: class_label = Variable(class_label)
    if class_out: class_idxs = Variable(class_idxs)

    # USING REAL LABEL TO UPDATE G
    binary_label = Variable(torch.ones(noise.size(0)))

    if cuda:
        noise = noise.cuda()
        binary_label = binary_label.cuda()
        if dual or class_out: class_label = class_label.cuda()
        if class_out: class_idxs = class_idxs.cuda()

    fake_image = G(noise, class_label)

    if dual:
        # binary_output, class_output = D(fake_image)
        # binary_output = binary_output.view(binary_output.numel())
        # class_output = class_output.view(batch_size, 10, 1, 1)

        # G_binary_loss = F.binary_cross_entropy(binary_output, binary_label)
        # G_class_loss = F.multilabel_soft_margin_loss(class_output, class_label)
        # G_loss = G_binary_loss + G_class_loss

        output = D(fake_image, class_label)
        output = output.view(output.numel())
        G_loss = F.binary_cross_entropy(output, binary_label)
    elif class_out:
        output = D(fake_image)
        output = output.view(noise.size(0), 11)
        print output
        print class_idxs.long()
        G_loss = F.cross_entropy(output, class_idxs.long())
    else:
        output = D(fake_image)
        output = output.view(output.numel())
        G_loss = F.binary_cross_entropy(output, binary_label)

    G_loss.backward()
    opt.step()
    if dual:
        return G_loss, G_loss
    else:
        return G_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gen_G_input(args, class_label = False, fake_image = False, target_class = None, batch_override = None):
    if batch_override:
        batch_size = batch_override
    else:
        batch_size = args.batch_size
    noise_dim = args.noise_dim

    noise = torch.FloatTensor(batch_size, noise_dim, 1, 1).normal_(0, 1)
    if class_label:
        if fake_image:
            class_tensor, class_idxs = gen_class_label(batch_size, vector_size = 11, target_class = 10, ret_c = True)
            return noise, class_tensor, class_idxs
        else:
            class_tensor = gen_class_label(batch_size, classes = 10, vector_size = 11, target_class = target_class)
            return noise, class_tensor, None
    else:
        return noise, None, None

        
def gen_class_label(batch_size, classes = 10, vector_size = None, target_class = None, ret_c = False):
    if target_class:
        c = [target_class for _ in range(batch_size)]
    else:
        c = [randint(0,classes - 1) for _ in range(batch_size)]

    if ret_c:
        c_out = torch.LongTensor(c)

    print 'C'
    print c_out

    if vector_size:
        class_label = torch.zeros(batch_size, vector_size, 1, 1)
    else:
        class_label = torch.zeros(batch_size, classes, 1, 1)
    for row, idx in enumerate(c):                
        class_label[row][idx] = 1
    if ret_c:
        return class_label, c_out
    else:
        return class_label

def save_samples(G, folder, filename, args, num_imgs = None):
    cuda = torch.cuda.is_available()
    noise_dim = args.noise_dim
    dual = args.dual_discrim
    class_out = args.class_output

    if num_imgs:
        batch_size = num_imgs
    else:    
        batch_size = args.batch_size

    inp = None

    if dual or class_out:
        for c in xrange(10):
            noise, class_label = gen_G_input(args, True, target_class = c, batch_override = 10)
            noise = Variable(noise)
            class_label = Variable(class_label)

            if cuda:
                noise = noise.cuda()
                class_label = class_label.cuda()

            fake_images = G(noise, class_label)
            utils.save_image(fake_images.data, '%s/%s_%d.png' % (folder, filename, c))
    else:
        inp, _, _ = gen_G_input(args)
        inp = Variable(inp)

        if cuda:
            inp = inp.cuda()

        fake_images = G(inp)
        utils.save_image(fake_images.data, '%s/%s.png' % (folder, filename))