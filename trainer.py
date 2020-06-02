import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.discriminator import discriminator
from models.generator import generator
from scripts.utils import Utils, Logger, from_onehot_to_int
from scripts.dataset_builder import dataset_builder, Rescale
from PIL import Image
import os
import numpy as np

class Trainer(object):
    def __init__(self, vis_screen, save_path, l1_coef, l2_coef, pre_trained_gen,
                 pre_trained_disc, batch_size, num_workers, epochs, inference, softmax_coef, image_size, lr_D, lr_G, audio_seconds):

        self.generator = generator(image_size, audio_seconds*16000).cuda()
        self.discriminator = discriminator(image_size).cuda()

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        self.inference = inference
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.softmax_coef = softmax_coef
        self.lr_D = lr_D
        self.lr_G = lr_G


        self.dataset = dataset_builder(transform=Rescale(int(self.image_size)), inference = self.inference, audio_seconds = audio_seconds)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)

        self.optimD = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.lr_D, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=self.lr_G, betas=(self.beta1, 0.999))

        self.logger = Logger(vis_screen, save_path)
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path

    def train(self):

        criterion = nn.MSELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        print('Training...')
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:

                right_images = sample['face']
                onehot = sample['onehot']
                raw_wav = sample['audio']
                wrong_images = sample['wrong_face']
                id_labels = from_onehot_to_int(onehot) 

                right_images = Variable(right_images.float()).cuda()
                raw_wav = Variable(raw_wav.float()).cuda()
                wrong_images = Variable(wrong_images.float()).cuda()
                onehot = Variable(onehot.float()).cuda()
                id_labels = Variable(id_labels).cuda()


                
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

               
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1)) # so smooth_real_labels will now be 0.9

                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                

                self.discriminator.zero_grad()

                fake_images, z_vector, _ = self.generator(raw_wav)

              
                outputs, _ = self.discriminator(fake_images, z_vector)

                fake_score = outputs 
                fake_loss = criterion(outputs, fake_labels)

                outputs, activation_real = self.discriminator(right_images, z_vector)

                real_score = outputs
                real_loss = criterion(outputs, smoothed_real_labels)

                outputs, _ = self.discriminator(wrong_images, z_vector)
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

                d_loss = real_loss + fake_loss + wrong_loss

                d_loss.backward()

                self.optimD.step()

                
                self.generator.zero_grad()

                fake_images, z_vector, softmax_scores = self.generator(raw_wav)

                outputs, activation_fake = self.discriminator(fake_images, z_vector)

                _, activation_real = self.discriminator(right_images, z_vector)


                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

               
                softmax_criterion = nn.CrossEntropyLoss()
                softmax_loss = softmax_criterion(softmax_scores, id_labels)


                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)\
                         + self.softmax_coef * softmax_loss  
                g_loss.backward()
                self.optimG.step()

            self.logger.log_iteration_gan(epoch, d_loss, g_loss, real_score, fake_score, wrong_score)

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

    def predict(self):
        print('Starting inference...')

        starting_id = 0 
        for id, sample in enumerate(self.data_loader):

            right_images = sample['face']
            onehot = sample['onehot']
            raw_wav = sample['audio']
            paths = sample['audio_path']

           
            token = (onehot == 1).nonzero()[:, 1]
            ids = [path.split('_')[-1][:-4] for path in paths]

            txt = [self.dataset.youtubers[idx] + '_' + str(id) for idx,id in zip(token,ids)]

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            raw_wav = Variable(raw_wav.float()).cuda()

            fake_images, _, _ = self.generator(raw_wav)

            for image, t in zip(fake_images, txt):
                im = image.data.mul_(127.5).add_(127.5).permute(1, 2, 0).cpu().numpy()
                rgb = np.empty((self.image_size, self.image_size, 3), dtype=np.float32)
                rgb[:,:,0] = im[:,:,2]
                rgb[:,:,1] = im[:,:,1]
                rgb[:,:,2] = im[:,:,0]
                im = Image.fromarray(rgb.astype('uint8'))
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))







