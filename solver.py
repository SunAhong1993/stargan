from x2paddle import torch2paddle
from model import Generator
from model import Discriminator
from paddle import to_tensor
from x2paddle.torch2paddle import save_image
import paddle
import paddle.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.test_iters = config.test_iters
        self.use_tensorboard = config.use_tensorboard
        self.device = 'cuda' if paddle.is_compiled_with_cuda() else 'cpu'
        self.device = self.device.replace('cuda', 'gpu')
        self.device = paddle.set_device(self.device)
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.
                c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 
                2, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.
                c_dim + self.c2_dim, self.d_repeat_num)
        self.g_optimizer = torch2paddle.Adam(self.G.parameters(), self.g_lr,
            [self.beta1, self.beta2])
        self.d_optimizer = torch2paddle.Adam(self.D.parameters(), self.d_lr,
            [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print('The number of parameters: {}'.format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters)
            )
        G_path = os.path.join(self.model_save_dir, '{}-G.pdiparams'.format(
            resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.pdiparams'.format(
            resume_iters))
        self.G.load_state_dict(paddle.load(G_path))
        self.D.load_state_dict(paddle.load(D_path))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = paddle.ones(y.size()).requires_grad_(False).to(self.device)
        dydx = paddle.grad(outputs=y, inputs=x, grad_outputs=weight,
            retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = paddle.sqrt(torch2paddle.sum(dydx ** 2, dim=1))
        return torch2paddle.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = paddle.zeros([batch_size, dim]).requires_grad_(False)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA',
        selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair',
                    'Gray_Hair']:
                    hair_color_indices.append(i)
        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg = c_trg.cast("int32")
                    c_trg_tmp = paddle.zeros_like(c_trg)
                    paddle.assign(c_trg, c_trg_tmp)
                    c_trg_tmp = c_trg_tmp.cast("bool")
                    c_trg_tmp[:, i] = c_trg[:, i] == 0
                    c_trg = c_trg_tmp 
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(paddle.ones(c_org.size(0)).
                    requires_grad_(False) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return torch2paddle.binary_cross_entropy_with_logits(logit,
                target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset,
            self.selected_attrs)
        g_lr = self.g_lr
        d_lr = self.d_lr
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            rand_idx = paddle.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device)
            c_trg = c_trg.to(self.device)
            label_org = label_org.to(self.device)
            label_trg = label_trg.to(self.device)
            out_src, out_cls = self.D(x_real)
            d_loss_real = -torch2paddle.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.
                dataset)
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch2paddle.mean(out_src)
            alpha = torch2paddle.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data
                ).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            d_loss = (d_loss_real + d_loss_fake + self.lambda_cls *
                d_loss_cls + self.lambda_gp * d_loss_gp)
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            if (i + 1) % self.n_critic == 0:
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = -torch2paddle.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg,
                    self.dataset)
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch2paddle.mean(paddle.abs(x_real - x_reconst))
                g_loss = (g_loss_fake + self.lambda_rec * g_loss_rec + self
                    .lambda_cls * g_loss_cls)
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = 'Elapsed [{}], Iteration [{}/{}]'.format(et, i + 1,
                    self.num_iters)
                for tag, value in loss.items():
                    log += ', {}: {:.4f}'.format(tag, value)
                print(log)
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)
            if (i + 1) % self.sample_step == 0:
                with paddle.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch2paddle.concat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir,
                        '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()),
                        sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(
                        sample_path))
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.pdiparams'
                    .format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.pdiparams'
                    .format(i + 1))
                paddle.save(self.G.state_dict(), G_path)
                paddle.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.
                    model_save_dir))
            if ((i + 1) % self.lr_update_step == 0 and i + 1 > self.
                num_iters - self.num_iters_decay):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(
                    g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA',
            self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = paddle.zeros([x_fixed.size(0), self.c_dim]
            ).requires_grad_(False).to(self.device)
        zero_rafd = paddle.zeros([x_fixed.size(0), self.c2_dim]
            ).requires_grad_(False).to(self.device)
        mask_celeba = self.label2onehot(paddle.zeros(x_fixed.size(0)).
            requires_grad_(False), 2).to(self.device)
        mask_rafd = self.label2onehot(paddle.ones(x_fixed.size(0)).
            requires_grad_(False), 2).to(self.device)
        g_lr = self.g_lr
        d_lr = self.d_lr
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)
                rand_idx = paddle.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]
                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = paddle.zeros([x_real.size(0), self.c2_dim]
                        ).requires_grad_(False)
                    mask = self.label2onehot(paddle.zeros(x_real.size(0)).
                        requires_grad_(False), 2)
                    c_org = torch2paddle.concat([c_org, zero, mask], dim=1)
                    c_trg = torch2paddle.concat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = paddle.zeros([x_real.size(0), self.c_dim]
                        ).requires_grad_(False)
                    mask = self.label2onehot(paddle.ones(x_real.size(0)).
                        requires_grad_(False), 2)
                    c_org = torch2paddle.concat([zero, c_org, mask], dim=1)
                    c_trg = torch2paddle.concat([zero, c_trg, mask], dim=1)
                x_real = x_real.to(self.device)
                c_org = c_org.to(self.device)
                c_trg = c_trg.to(self.device)
                label_org = label_org.to(self.device)
                label_trg = label_trg.to(self.device)
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim
                    ] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = -torch2paddle.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org,
                    dataset)
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch2paddle.mean(out_src)
                alpha = torch2paddle.rand(x_real.size(0), 1, 1, 1).to(self.
                    device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data
                    ).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)
                d_loss = (d_loss_real + d_loss_fake + self.lambda_cls *
                    d_loss_cls + self.lambda_gp * d_loss_gp)
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                if (i + 1) % self.n_critic == 0:
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim
                        ] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = -torch2paddle.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls,
                        label_trg, dataset)
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch2paddle.mean(paddle.abs(x_real -
                        x_reconst))
                    g_loss = (g_loss_fake + self.lambda_rec * g_loss_rec + 
                        self.lambda_cls * g_loss_cls)
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = ('Elapsed [{}], Iteration [{}/{}], Dataset [{}]'.
                        format(et, i + 1, self.num_iters, dataset))
                    for tag, value in loss.items():
                        log += ', {}: {:.4f}'.format(tag, value)
                    print(log)
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i + 1)
            if (i + 1) % self.sample_step == 0:
                with paddle.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch2paddle.concat([c_fixed, zero_rafd,
                            mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch2paddle.concat([zero_celeba, c_fixed,
                            mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch2paddle.concat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir,
                        '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()),
                        sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(
                        sample_path))
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.pdiparams'
                    .format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.pdiparams'
                    .format(i + 1))
                paddle.save(self.G.state_dict(), G_path)
                paddle.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.
                    model_save_dir))
            if ((i + 1) % self.lr_update_step == 0 and i + 1 > self.
                num_iters - self.num_iters_decay):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(
                    g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        self.restore_model(self.test_iters)
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        with paddle.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.
                    dataset, self.selected_attrs)
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))
                x_concat = torch2paddle.concat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'
                    .format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path,
                    nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(
                    result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        self.restore_model(self.test_iters)
        with paddle.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim,
                    'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = paddle.zeros([x_real.size(0), self.c_dim]
                    ).requires_grad_(False).to(self.device)
                zero_rafd = paddle.zeros([x_real.size(0), self.c2_dim]
                    ).requires_grad_(False).to(self.device)
                mask_celeba = self.label2onehot(paddle.zeros(x_real.size(0)
                    ).requires_grad_(False), 2).to(self.device)
                mask_rafd = self.label2onehot(paddle.ones(x_real.size(0)).
                    requires_grad_(False), 2).to(self.device)
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch2paddle.concat([c_celeba, zero_rafd,
                        mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch2paddle.concat([zero_celeba, c_rafd,
                        mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                x_concat = torch2paddle.concat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'
                    .format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path,
                    nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(
                    result_path))
