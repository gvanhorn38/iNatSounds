import os
import tqdm
import librosa
import numpy as np
import soundfile as sf
import argparse
import cv2
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse
import os
import tqdm
import cv2
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy import signal
import PIL
import matplotlib.pyplot as plt 
import json

"""
Numpy/Scipy/PIL Version

Utilities to convert audio into an image. 
If time, frequency boxes are provided, then they will be mapped onto the generated image. 
"""




# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def mel_to_hertz(mel_values):
  """Converts frequencies in `mel_values` from the mel scale to linear scale.
  Adapted from tensorflow/python/ops/signal/mel_ops.py
  """
  return _MEL_BREAK_FREQUENCY_HERTZ * (
      np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0)


def hertz_to_mel(frequencies_hertz):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.
  Adapted from tensorflow/python/ops/signal/mel_ops.py
  """
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hertz_to_mel(fmin)
    max_mel = hertz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hertz(mels) 

def validate_mel_arguments(num_mel_bins, num_spectrogram_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz):
    
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            'num_spectrogram_bins must be positive. Got: %s' % num_spectrogram_bins)
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
        'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                     % (upper_edge_hertz, sample_rate))

def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=16000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
    Adapted from tf.signal.linear_to_mel_weight_matrix, found in tensorflow/python/ops/signal/mel_ops.py

    Args:
    num_mel_bins: Int, number of output frequency dimensions.
    num_spectrogram_bins: Int, number of input frequency dimensions.
    sample_rate: Int, sample rate of the audio.
    lower_edge_hertz: Float, lowest frequency to consider.
    upper_edge_hertz: Float, highest frequency to consider.

    Returns:
    Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].

    Raises:
    ValueError: Input argument in the wrong range.
    """
    
    validate_mel_arguments(num_mel_bins, num_spectrogram_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
    spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), 
        hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2
    )

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]
    
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)
    
    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    
    # Re-add the zeroed lower bins we sliced out above.
    return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], 'constant')
    

class AudioToImageConverter(object):
    
    def __init__(self,
        freq_scale='linear',
        samplerate=22050,
        fft_length=1024,
        window_length_samples=256,
        hop_length_samples=32,
        mel_bands=298,
        mel_min_hz=50,
        mel_max_hz=11025,
        amin=1e-10,
        ref_power_value=1.,
        max_db_value=0.,
        min_db_value=-100.,
        target_height=None,
        target_width=None,
        save_distribution=None
    ):
    
        self.freq_scale = freq_scale

        # STFT Configurations
        self.samplerate = samplerate
        self.fft_length = fft_length
        self.window_length_samples = window_length_samples
        self.hop_length_samples = hop_length_samples

        # Mel Scaling Configurations
        self.mel_bands = mel_bands
        self.mel_min_hz = mel_min_hz
        self.mel_max_hz = mel_max_hz

        # dB Conversion Configurations
        self.amin = amin
        self.ref_power_value = ref_power_value
        self.max_db_value = max_db_value
        self.min_db_value = min_db_value

        # Resizing Configurations
        self.target_height = target_height
        self.target_width = target_width

        if self.freq_scale == 'mel':
            self.mel_frequencies = mel_frequencies(n_mels=self.mel_bands + 1, fmin=self.mel_min_hz, fmax=self.mel_max_hz)

        self.global_min = None
        self.global_max = None
        self.bins = list(range(-200, 100))
        self.db_dist = {
            bin: 0 for bin in self.bins
        }
        self.save_distribution = save_distribution
            
    
    def _samples_to_magnitude_spectrogram(self, samples):
        """ Generate the magnitude spectrogram. 
        Input:
          samples: float32, assumed to be in the range [-1, 1] 
          (e.g. int16 values from a 16bit WAV file -> convert to float32 -> divide by 32767.0)
        Output:
          [T, F] matrix with T time bins and F frequency bins with values in the range [0, 1]
        """

        # Compute the Fourier spectrum on overlapping windows to create a spectrogram        
        window = signal.windows.hann(self.window_length_samples, sym=False)
        frequencies, times, spectrum = signal.stft(
            samples,
            fs=1,
            window=window,
            nperseg=self.window_length_samples,
            noverlap=self.window_length_samples - self.hop_length_samples,
            nfft=self.fft_length,
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
            scaling='spectrum',
        )
        if len(spectrum.shape) < 2:
            spectrum = np.zeros((self.fft_length, self.window_length_samples))
        
        # NOTE: scipy.signal.stft rescales the output by np.sqrt(1. / window.sum() ** 2). 
        # https://github.com/scipy/scipy/blob/43b76937c0aadf23b4e0fcc04f546fa5d61b543b/scipy/signal/spectral.py#L1805
        # Undo this scaling 
        scale = np.sqrt(1.0 / window.sum()**2)
        spectrum = spectrum / scale

        # Compute the magnitude values:
        # 1. Take the abosulte value to get the magnitude of the complex values in the spectrum
        # 2. Scale by a factor of 2 since we are using half the FFT spectrum
        # 3. Scale by the window values to compensate for the loss of energy due to multiplication by that window
        magnitudes = np.abs(spectrum) * 2. / window.sum()
        
        # magnitudes = [Frequency, Time] 
        # frequencies go from low to high
        # We will follow the TensorFlow format and convert this to [Time, Frequency] to maintain compatibility between the methods
        magnitudes = np.transpose(magnitudes)
        
        return magnitudes
    
    def _mel_scale(self, spectrogram):
        """Apply Mel scaling to the frequency magnitudes."""

        weight_matrix = linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bands,
            num_spectrogram_bins=spectrogram.shape[1],
            sample_rate=self.samplerate,
            lower_edge_hertz=self.mel_min_hz,
            upper_edge_hertz=self.mel_max_hz
        )
        spectrogram = np.matmul(spectrogram, weight_matrix)

        return spectrogram
    
    def _magnitude_to_db(self, spectrogram):
        """ Convert a magnitude spectorgram to a dB spectrogram.
        Input:
          spectrogram: [T, F] matrix of magnitude values
        output:
          spectrogram: [T, F] matrix of dB values
        """

        # Take 2 * 10 * log10 [the factor of two comes from taking the power of the magnitudes]
        # i.e. spectrogram = 10 * log10(tf.math.pow(spectrogram, 2.0)) -> 20. * log10(spectrogram)
        # NOTE: we can't take a log of 0, so clamp the values to something > 0
        log_spectrogram = 20.0 * np.log10(np.maximum(self.amin, spectrogram))

        # Convert to dB by subtracting off the reference power value.
        # 0s in the dB spectrogram will correspond to the reference power value.
        # (again make sure we don't take log10(0) )
        db_spectrogram = log_spectrogram - 20.0 * np.log10(np.maximum(self.amin, self.ref_power_value))

        # Clamp the dB values to be in a specific range
        mx, mn = db_spectrogram.max(), db_spectrogram.min()
        self.global_max = mx if self.global_max is None else max(mx, self.global_max)
        self.global_min = mn if self.global_min is None else min(mn, self.global_min)
        # self.db_dist.extend(db_spectrogram.flatten().tolist())
        # print(self.db_dist)
        if self.save_distribution is not None: 
            for k in range(len(self.bins) - 1):
                lower = self.bins[k]
                upper = self.bins[k+1]
                self.db_dist[lower] += int(np.sum((db_spectrogram >= lower) * (db_spectrogram < upper)))
        


        # print(self.global_max, self.global_min)
        # print(db_spectrogram.max(), db_spectrogram.min())
        db_spectrogram = np.clip(db_spectrogram, self.min_db_value, self.max_db_value)
        

        return db_spectrogram
    
    def _db_to_uint8(self, spectrogram):
        """Scale the dB values to be in the range [0, 255] and cast to uint8"""
        spectrogram = ((spectrogram - self.min_db_value) / (self.max_db_value - self.min_db_value)) * 255.
        spectrogram = spectrogram.astype(np.uint8)
        return spectrogram

    def _orientate(self, spectrogram):
        """Rotate and transpose the [T, F] spectrogram to be [F, T] with frequency rows going from high to low, 
        and time columns going from beginning to end.
        """
        spectrogram = np.transpose(spectrogram)
        spectrogram = np.flip(spectrogram, axis=[0])
        return spectrogram
    
    def _resize(self, spectrogram):

        freq_scale = 1.0
        time_scale = 1.0
        spectrogram_shape = spectrogram.shape

        # Did we specify both a height and width? 
        if self.target_height is not None and self.target_width is not None:
            
            # `order` details: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp
            #spectrogram = skimage.transform.resize(spectrogram, [self.target_height, self.target_width], order=1, preserve_range=True)
            spectrogram = np.array(PIL.Image.fromarray(spectrogram).resize([self.target_width, self.target_height], resample=PIL.Image.BILINEAR))
            freq_scale = float(self.target_height) / spectrogram_shape[0]
            time_scale = float(self.target_width) / spectrogram_shape[1]

        # Did we just specify a height?
        elif self.target_height is not None:

            frac = float(spectrogram_shape[0]) / self.target_height
            width_target = int(float(spectrogram_shape[1]) / frac)
            
            spectrogram = np.array(PIL.Image.fromarray(spectrogram).resize([width_target, self.target_height], resample=PIL.Image.BILINEAR))

            freq_scale = float(self.target_height) / spectrogram_shape[0]
            time_scale = float(width_target) / spectrogram_shape[1]

        # Did we just specify a width?
        elif self.target_width is not None:

            frac = float(spectrogram_shape[1]) / self.target_width
            height_target = int(float(spectrogram_shape[0]) / frac)
            
            spectrogram = np.array(PIL.Image.fromarray(spectrogram).resize([self.target_width, height_target], resample=PIL.Image.BILINEAR))

            freq_scale = float(height_target) / spectrogram_shape[0]
            time_scale = float(self.target_width) / spectrogram_shape[1]

        return spectrogram, freq_scale, time_scale
    
    def _transform_boxes(self, spectrogram, boxes, freq_scale=1.0, time_scale=1.0):
        """ Convert time/freq boxes to normalized spectrogram coordinates.

        Spectrogram is assumed to be in the [F, T] format, with frequencies going from "high" to "low" with increasing y value (i.e assuming the origin is the top left corner). 
        
        Boxes are in the format:
        np.array([
            [high_freq, start_sec, low_freq, end_sec],
            ...
        ])
        
        Transformed boxes are in normalized coordinates with the following format: 
        np.array([
            [ymin, xmin, ymax, xmax],
            ...
        ])
        """

        spectrogram_shape = spectrogram.shape
        spectrogram_height_float = float(spectrogram_shape[0])
        spectrogram_width_float = float(spectrogram_shape[1])
    
        samplerate_float = float(self.samplerate)
        hop_length_float = float(self.hop_length_samples)

        # Conversion factor that will take us from `seconds` to `time bin` in the spectrogram
        sec_to_time_bin = (samplerate_float / hop_length_float) * time_scale
        spec_start_sec = boxes[:,1] * sec_to_time_bin
        spec_end_sec = boxes[:,3] * sec_to_time_bin

        # Need to map "frequency" to the corresponding frequency bin in the spectrogram
        # Need to account for the change in coordinate system to the upper left corner
        if self.freq_scale == 'mel':
            spec_high_freq =  spectrogram_height_float - ((np.digitize(boxes[:,0], bins=self.mel_frequencies, right=False) - 1) * freq_scale)
            spec_low_freq = spectrogram_height_float - ((np.digitize(boxes[:,2], bins=self.mel_frequencies, right=False) - 1) * freq_scale)

        else:
            x_ref = np.linspace(start=0., stop=self.samplerate // 2, num=self.fft_length // 2)
            y_ref = np.linspace(start=0., stop=self.fft_length // 2, num=self.fft_length // 2)
            spec_high_freq =  spectrogram_height_float - (np.interp(boxes[:,0], x_ref, y_ref) * freq_scale) 
            spec_low_freq = spectrogram_height_float - (np.interp(boxes[:,2], x_ref, y_ref) * freq_scale) 

        # Clip the values to ensure they are in range
        spec_start_sec = np.clip(spec_start_sec, 0., spectrogram_width_float - 1 )
        spec_end_sec = np.clip(spec_end_sec, 0., spectrogram_width_float - 1 )
        spec_high_freq = np.clip(spec_high_freq, 0., spectrogram_height_float - 1 )
        spec_low_freq = np.clip(spec_low_freq, 0., spectrogram_height_float - 1 )

        # Return normalized coordinates 
        spec_boxes = np.stack([
            spec_high_freq / spectrogram_height_float,
            spec_start_sec / spectrogram_width_float,
            spec_low_freq / spectrogram_height_float,
            spec_end_sec / spectrogram_width_float
        ], axis=-1)

        return spec_boxes

    def __call__(self, samples, boxes=None):
        """ Create a spectrogram image from the samples and transform the boxes so that 
        they are in normalized image coordinates. 
        
        samples: [float32], assumed to be mono channel and in the range [-1, 1] 
          (e.g. int16 values from a 16bit WAV file -> convert to float32 -> divide by 32767.0)
        
        boxes: [[float32]]
        
        Boxes are in the format:
        np.array([
            [high_freq, start_sec, low_freq, end_sec],
            ...
        ])
        
        """
        
        spectrogram = self._samples_to_magnitude_spectrogram(samples) # float32
        if self.freq_scale == 'mel':
            spectrogram = self._mel_scale(spectrogram) # float32
        spectrogram = self._magnitude_to_db(spectrogram) # float32
        spectrogram = self._db_to_uint8(spectrogram) # uint8
        spectrogram = self._orientate(spectrogram) # uint8
        spectrogram, freq_scale, time_scale = self._resize(spectrogram) # uint8
        if self.save_distribution is not None: 
            with open(self.save_distribution, "w") as f:
                json.dump(self.db_dist, f)
        if boxes is not None and boxes.shape[0]:
            spectrogram_boxes = self._transform_boxes(spectrogram, boxes, freq_scale, time_scale)
            return spectrogram, spectrogram_boxes
        else:
            return spectrogram


def convert_wav2img(wav_path, np_path, vis_path, wav2spec):

    sr, waveform = scipy.io.wavfile.read(wav_path)
    if len(waveform.shape) > 1:
        waveform = waveform[:, 0]
    spectrogram = wav2spec(waveform)

    np.save(np_path, spectrogram)
    # cv2.imwrite(vis_path, spectrogram)

    return spectrogram.shape[-1]
    
    

def convert_file_list(args, file_list, wav2spec):
    for dirname, audio_name in tqdm.tqdm(file_list, leave=False):
        np_name = audio_name.replace(".wav", ".npy")
        vis_name = audio_name.replace(".wav", ".jpg")

        wav_path = os.path.join(args.dir, dirname, audio_name)
        np_path_fmt = os.path.join(args.np_dir, dirname, np_name)
        vis_path_fmt = os.path.join(args.vis_dir, dirname, vis_name)

        try:
            ran = convert_wav2img(wav_path, np_path_fmt, vis_path_fmt, wav2spec)
        except KeyboardInterrupt:
            exit()
        except:
            print("Error!!", dirname, audio_name)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./sound_files/sound_wav_22050")
    parser.add_argument('--np_dir', type=str, default="./mel_np_full")
    parser.add_argument('--vis_dir', type=str, default="./mel_jpg")
    args = parser.parse_args()

    os.makedirs(args.np_dir, exist_ok=True)
    # os.makedirs(args.vis_dir, exist_ok=True)

    dirlist = [(os.path.join(args.dir, d), d) for d in os.listdir(args.dir)]

    for _, d in dirlist:
        os.makedirs(os.path.join(args.np_dir, d), exist_ok=True)
        # os.makedirs(os.path.join(args.vis_dir, d), exist_ok=True)

    dirlist = sorted(dirlist)
    file_list = []
    for dir_path, dirname in dirlist:
        if ".DS" in dirname: continue
        for fname in os.listdir(dir_path):
            if ".DS" in fname: continue

            file_list.append(
                (dirname, fname)
            )

    file_list = sorted(file_list)

    wav2spec = AudioToImageConverter(
        freq_scale='mel',
        samplerate=22050,
        fft_length=1024,
        window_length_samples=512,
        hop_length_samples=128,
        mel_bands=128,
        mel_min_hz=0,
        mel_max_hz=11025,
        amin=1e-10,
        ref_power_value=1.,
        max_db_value=50.,
        min_db_value=-50.,
        target_height=None,
        target_width=None,
        save_distribution=None
    )


    n_threads = 64
    processes = []
    size_each = (len(file_list) + n_threads - 1) // n_threads
    for rank in range(n_threads):
        strt = rank*size_each
        end = min((rank+1)*size_each, len(file_list))
        entries = file_list[strt:end]
        p = Process(target=convert_file_list, args=(args, entries, wav2spec))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    