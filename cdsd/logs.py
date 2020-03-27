from csv import DictWriter


class CDSDHistoryLogger(object):
    def __init__(self, path):
        self.path = path
        self.field_names = ["epoch", "train_mix_loss", "train_cls_loss", "train_tot_loss",
                            "valid_mix_loss", "valid_cls_loss", "valid_tot_loss"]
        # Use write mode first instead of append to clear any existing files
        with open(self.path, 'w') as f:
            writer = DictWriter(f, fieldnames=self.field_names)
            writer.writeheader()

        self.best_valid_loss = None
        self.best_valid_loss_flag = False

    def log(self, epoch, train_mix_loss, train_cls_loss, train_tot_loss,
            valid_mix_loss, valid_cls_loss, valid_tot_loss):
        with open(self.path, 'a') as f:
            writer = DictWriter(f, fieldnames=self.field_names)
            writer.writerow({
                "epoch": epoch,
                "train_mix_loss": train_mix_loss,
                "train_cls_loss": train_cls_loss,
                "train_tot_loss": train_tot_loss,
                "valid_mix_loss": valid_mix_loss,
                "valid_cls_loss": valid_cls_loss,
                "valid_tot_loss": valid_tot_loss
            })

        if self.best_valid_loss is None or self.best_valid_loss > valid_tot_loss:
            self.best_valid_loss = valid_tot_loss
            self.best_valid_loss_flag = True
        else:
            self.best_valid_loss_flag = False

    def valid_loss_improved(self):
        # SHOULD ONLY BE CALLED AFTER CALLING *.log()
        return self.best_valid_loss_flag


class ClassifierHistoryLogger(object):
    def __init__(self, path):
        self.path = path
        self.field_names = ["epoch", "train_loss", "valid_loss"]
        # Use write mode first instead of append to clear any existing files
        with open(self.path, 'w') as f:
            writer = DictWriter(f, fieldnames=self.field_names)
            writer.writeheader()

        self.best_valid_loss = None
        self.best_valid_loss_flag = False

    def log(self, epoch, train_loss, valid_loss):
        with open(self.path, 'a') as f:
            writer = DictWriter(f, fieldnames=self.field_names)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            })

        if self.best_valid_loss is None or self.best_valid_loss > valid_loss:
            self.best_valid_loss = valid_loss
            self.best_valid_loss_flag = True
        else:
            self.best_valid_loss_flag = False

    def valid_loss_improved(self):
        # SHOULD ONLY BE CALLED AFTER CALLING *.log()
        return self.best_valid_loss_flag
