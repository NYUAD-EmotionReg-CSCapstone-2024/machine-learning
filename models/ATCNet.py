from braindecode.models import ATCNet as BaseATCNet

class ATCNet(BaseATCNet):
    def __init__(self, n_chans=62, n_classes=5, input_window_seconds=4, sfreq=200):
        super(ATCNet, self).__init__(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            tcn_kernel_size=5,
            add_log_softmax=False
        )