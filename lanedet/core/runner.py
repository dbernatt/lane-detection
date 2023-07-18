from lanedet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from lanedet.datasets.registry import build_dataloader


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = build_net(self.cfg)
        self.optimizer = build_optimizer(self.cfg, self.net)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            output = self.net(data)

            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()

            # if i % self.cfg.log_interval == 0 or i == max_iter - 1:
            #     lr = self.optimizer.param_groups[0]['lr']

    def train(self):
        print('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        print('Start training...')
        start_epoch = 0

        for epoch in range(start_epoch, self.cfg.epochs):
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()
