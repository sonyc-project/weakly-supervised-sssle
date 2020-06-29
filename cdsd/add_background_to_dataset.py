import os
import random
import shutil
import sys
import tempfile
import jams
import sox
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from scaper.audio import get_integrated_lufs, match_sample_length


def add_background(fg_dir, bg_dir, out_dir, ref_db, exp_label, random_state):
    random.seed(random_state)
    os.makedirs(out_dir, exist_ok=True)

    for fg_fname in os.listdir(fg_dir):
        if not fg_fname.endswith('.wav'):
            continue

        # Set up input and output paths
        fg_audio_path = os.path.join(fg_dir, fg_fname)
        fg_jams_path = os.path.splitext(fg_audio_path)[0] + '.jams'
        fg_events_dir = os.path.splitext(fg_audio_path)[0] + '_events'

        # Get experiment label so we can replace it if necessary
        l_idx = fg_fname.index('_')
        r_idx = fg_fname.rindex('_')
        if l_idx == r_idx:
            l_idx = 0
        else:
            l_idx += 1
        orig_exp_label = fg_fname[l_idx:r_idx]
        if not exp_label:
            exp_label = orig_exp_label
        out_audio_path = os.path.join(out_dir, os.path.basename(fg_audio_path).replace(orig_exp_label, exp_label))
        out_jams_path = os.path.join(out_dir, os.path.basename(fg_jams_path).replace(orig_exp_label, exp_label))
        out_events_dir = os.path.splitext(out_audio_path)[0] + '_events'
        os.makedirs(out_events_dir, exist_ok=True)

        # Load JAMS file
        jam = jams.load(fg_jams_path)
        ann = jam.annotations[0]
        duration = ann.duration
        # Check that revereb is disabled
        if 'scaper' in ann.sandbox and 'reverb' in ann.sandbox['scaper'] and \
                ann.sandbox['scaper']['reverb']:
            raise ValueError('Reverb should not be enabled.')
        if 'scaper' in ann.sandbox and 'sample_rate' in ann.sandbox['scaper']:
            sr = ann.sandbox['scaper']['sample_rate']
        else:
            sr = 16000

        # Choose a background file and start time
        background_label = random.choice(os.listdir(bg_dir))
        bg_fname = random.choice(os.listdir(os.path.join(bg_dir, background_label)))
        bg_audio_path = os.path.join(bg_dir, background_label, bg_fname)
        source_duration = (sox.file_info.duration(bg_audio_path))
        ntiles = int(max(duration // source_duration + 1, 2))
        bg_time = random.random() * max(duration - source_duration, 0)

        bg_event_audio_path = os.path.join(
            out_events_dir,
            'background0_{:s}{:s}'.format(background_label, '.wav'))


        ## Update JAMS file and save
        # Update SNR of sources
        for obs in jam.annotations[0].data:
            obs.value['snr'] -= ref_db
        # Add background event
        ann.append(time=0.0,
                   duration=duration,
                   value={
                       "label": "background",
                       "source_file": bg_audio_path,
                       "source_time": bg_time,
                       "event_time": 0.0,
                       "event_duration": duration,
                       "snr": 0.0,
                       "role": "background",
                       "pitch_shift": None,
                       "time_stretch": None
                   },
                   confidence=1.0)
        jam.save(out_jams_path)

        # Copy events
        for event_fname in os.listdir(fg_events_dir):
            src_event_audio_path = os.path.join(fg_events_dir, event_fname)
            dst_event_audio_path = os.path.join(out_events_dir, event_fname)
            shutil.copy(src_event_audio_path, dst_event_audio_path)

        # Process background by trimming/concat'ing it and adjusting gain
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f_bg_temp:
            # Create combiner
            cmb = sox.Combiner()
            cmb.set_input_format(file_type=['wav', 'wav'])
            # Ensure consistent sampling rate and channels
            cmb.convert(samplerate=sr, n_channels=1, bitdepth=None)
            # Then trim the duration of the background event
            cmb.trim(bg_time, bg_time + duration)
            # synthesize concatenated/trimmed background
            cmb.build([bg_audio_path] * ntiles, f_bg_temp.name, 'concatenate')

            # NOW compute LUFS
            bg_lufs = get_integrated_lufs(f_bg_temp.name)
            # Normalize background to reference DB.
            gain = ref_db - bg_lufs
            # Use transformer to adapt gain
            tfm = sox.Transformer()
            tfm.gain(gain_db=gain, normalize=False)
            tfm.build(f_bg_temp.name, bg_event_audio_path)

        # Mix foreground and background together
        cmb = sox.Combiner()
        cmb.set_input_format(file_type=['wav', 'wav'])
        cmb.build([fg_audio_path, bg_event_audio_path], out_audio_path, 'mix')

        # Make sure every single audio file has exactly the same duration
        # using soundfile.
        duration_in_samples = int(duration * sr)
        match_sample_length(out_audio_path, duration_in_samples)
        match_sample_length(fg_audio_path, duration_in_samples)
        match_sample_length(bg_event_audio_path, duration_in_samples)


def parse_arguments(args):
    parser = ArgumentParser(sys.argv[0],
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('fg_folder', type=str,
                        help='Path to foreground audio directory')
    parser.add_argument('bg_folder', type=str,
                        help='Path to background audio directory')
    parser.add_argument('out_folder', type=str,
                        help='Output directory where audio files will be saved.')
    parser.add_argument('ref_db', type=float,
                        help='Background loudness (in LUFS).')
    parser.add_argument('exp_label', type=str,
                        help='Label to use for this set of generated audio clips.')
    parser.add_argument('--random-state', type=int, default=123,
                        help='Random seed to set for reproducability.')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    add_background(fg_dir=args.fg_folder,
                   bg_dir=args.bg_folder,
                   out_dir=args.out_folder,
                   ref_db=args.ref_db,
                   exp_label=args.exp_label,
                   random_state=args.random_state)

