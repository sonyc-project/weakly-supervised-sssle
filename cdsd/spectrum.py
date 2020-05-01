import torch, torchaudio
from torchaudio.functional import spectrogram as ta_spectrogram
from torchaudio.functional import istft as ta_istft
from torchaudio.transforms import Spectrogram as TorchAudioSpectrogram
from torchaudio.transforms import MelSpectrogram as TorchAudioMelSpectrogram


def spectrogram(waveform, pad, window, n_fft, hop_length, win_length, power, normalized, window_scaling):
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.

    Args:
        waveform (torch.Tensor): Tensor of audio of dimension (..., time)
        pad (int): Two sided padding of signal
        window (torch.Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool): Whether to normalize by magnitude after stft
        window_scaling (bool): Whether to scale by sum of window function

    Returns:
        torch.Tensor: Dimension (..., freq, time), freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """

    spec = ta_spectrogram(waveform, pad, window, n_fft, hop_length, win_length, power, normalized)
    if window_scaling:
        spec /= window.sum().clone().detach().type(spec.dtype).to(spec.device)

    return spec


def istft(stft_matrix, n_fft, hop_length=None, win_length=None, window=None,
          center=True, pad_mode="reflect", normalized=False,
          window_scaling=True, onesided=True, length=None):
    r"""Inverse short time Fourier Transform. This is expected to be the inverse of torch.stft.
    It has the same parameters (+ additional optional parameter of ``length``) and it should return the
    least squares estimation of the original signal. The algorithm will check using the NOLA condition (
    nonzero overlap).
    Important consideration in the parameters ``window`` and ``center`` so that the envelop
    created by the summation of all the windows is never zero at certain point in time. Specifically,
    :math:`\sum_{t=-\infty}^{\infty} w^2[n-t\times hop\_length] \cancel{=} 0`.
    Since stft discards elements at the end of the signal if they do not fit in a frame, the
    istft may return a shorter signal than the original signal (can occur if ``center`` is False
    since the signal isn't padded).
    If ``center`` is True, then there will be padding e.g. 'constant', 'reflect', etc. Left padding
    can be trimmed off exactly because they can be calculated but right padding cannot be calculated
    without additional information.
    Example: Suppose the last window is:
    [17, 18, 0, 0, 0] vs [18, 0, 0, 0, 0]
    The n_frame, hop_length, win_length are all the same which prevents the calculation of right padding.
    These additional values could be zeros or a reflection of the signal so providing ``length``
    could be useful. If ``length`` is ``None`` then padding will be aggressively removed
    (some loss of signal).
    [1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
    IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.
    Args:
        stft_matrix (Tensor): Output of stft where each row of a channel is a frequency and each
            column is a window. It has a size of either (..., fft_size, n_frame, 2)
        n_fft (int): Size of Fourier transform
        hop_length (int or None, optional): The distance between neighboring sliding window frames.
            (Default: ``win_length // 4``)
        win_length (int or None, optional): The size of window frame and STFT filter. (Default: ``n_fft``)
        window (Tensor or None, optional): The optional window function.
            (Default: ``torch.ones(win_length)``)
        center (bool, optional): Whether ``input`` was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (str, optional): Controls the padding method used when ``center`` is True. (Default:
            ``"reflect"``)
        normalized (bool, optional): Whether the STFT was normalized. (Default: ``False``)
        window_scaling (bool, optional): Whether the STFT was scaled by the sum of the window function. (Default: ``False``)
        onesided (bool, optional): Whether the STFT is onesided. (Default: ``True``)
        length (int or None, optional): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)
    Returns:
        Tensor: Least squares estimation of the original signal of size (..., signal_length)
    """
    inv_spec = ta_istft(stft_matrix, n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window,
                        center=center,
                        pad_mode=pad_mode,
                        normalized=normalized,
                        onesided=onesided,
                        length=length)

    if window_scaling:
        # Scale output by sum of window
        if window is None:
            # Default window is rectangular
            scale = win_length or n_fft
        else:
            scale = window.sum()
        scale = scale.clone().detach().type(inv_spec.dtype).to(inv_spec.device)
        inv_spec *= scale

    return inv_spec


class Spectrogram(TorchAudioSpectrogram):
    def __init__(self, n_fft=400, win_length=None, hop_length=None, pad=0,
                 window_fn=torch.hann_window, power=2.0, normalized=False,
                 window_scaling=False, wkwargs=None):
        super(Spectrogram, self).__init__(n_fft=n_fft,
                                          win_length=win_length,
                                          hop_length=hop_length,
                                          pad=pad,
                                          window_fn=window_fn,
                                          power=power,
                                          normalized=normalized,
                                          wkwargs=wkwargs)
        self.window_scaling = window_scaling

    def forward(self, waveform):
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return spectrogram(waveform, self.pad, self.window, self.n_fft,
                           self.hop_length, self.win_length, self.power,
                           self.normalized, self.window_scaling)


class MelSpectrogram(TorchAudioMelSpectrogram):
    r"""Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
    and MelScale.
    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool, optional): Whether the STFT was normalized. (Default: ``False``)
        window_scaling (bool, optional): Whether the STFT was scaled by the sum of the window function. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalization=True)
        >>> mel_specgram = MelSpectrogram(sample_rate)(waveform)  # (channel, n_mels, time)
    """
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None,
                 hop_length=None, f_min=0., f_max=None, pad=0, n_mels=128,
                 window_fn=torch.hann_window, power=2., normalized=False,
                 window_scaling=False, wkwargs=None):
        super(MelSpectrogram, self).__init__(sample_rate=sample_rate,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             f_min=f_min,
                                             f_max=f_max,
                                             pad=pad,
                                             n_mels=n_mels,
                                             window_fn=window_fn,
                                             wkwargs=wkwargs)
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=power,
                                       normalized=normalized, window_scaling=window_scaling,
                                       wkwargs=wkwargs)







