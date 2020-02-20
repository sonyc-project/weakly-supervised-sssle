from csv import DictWriter


class HistoryLogger(object):
    def __init__(self, path):
        self.file = open(path, 'w')
        field_names = ["epoch", "train_mix_loss", "train_cls_loss", "train_tot_loss",
                       "valid_mix_loss", "valid_cls_loss", "valid_tot_loss"]
        self.writer = DictWriter(self.file, fieldnames=field_names)
        self.writer.writeheader()

        self.best_valid_loss = None
        self.best_valid_loss_flag = False

    def log(self, epoch, train_mix_loss, train_cls_loss, train_tot_loss,
            valid_mix_loss, valid_cls_loss, valid_tot_loss):
        self.writer.writerow({
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

    def close(self):
        self.file.close()
