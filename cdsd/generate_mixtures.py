import json
import os
import random
import sys
import numpy as np
import scaper
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from numbers import Number
from tqdm import tqdm


def positive_int(value):
    """An argparse type method for accepting only positive ints"""
    try:
        ivalue = int(value)
    except (ValueError, TypeError) as e:
        raise ArgumentTypeError('Expected a positive int, error message: '
                                '{}'.format(e))
    if ivalue <= 0:
        raise ArgumentTypeError('Expected a positive int')
    return ivalue


def get_distribution_tuple(dist):
    """Return distribution tuple from a distribution dictionary"""
    if not dist:
        return None

    dist_type = dist['dist']

    if dist_type == 'const':
        return ('const', dist['value'])
    elif dist_type == 'choose':
        valuelist = dist['valuelist']
        assert type(valuelist) == list
        return ('choose', valuelist)
    elif dist_type == 'uniform':
        min = dist['min']
        max = dist['max']
        assert isinstance(min, Number)
        assert isinstance(max, Number)
        return ('uniform', min, max)
    elif dist_type == 'normal':
        mean = dist['mean']
        std = dist['std']
        assert isinstance(mean, Number)
        assert isinstance(std, Number)
        return ('normal', mean, std)
    elif dist_type == 'truncnorm':
        mean = dist['mean']
        std = dist['std']
        min = dist['min']
        max = dist['max']
        assert isinstance(mean, Number)
        assert isinstance(std, Number)
        assert isinstance(min, Number)
        assert isinstance(max, Number)
        return ('truncnorm', mean, std, min, max)
    else:
        raise ValueError('Invalid distribution type: {}'.format(dist_type))


def sample_ztpoisson(lmbda):
    """
    Samples a value from the zero-truncated poisson distrubution, using the
    algorithm described in http://giocc.com/zero_truncated_poisson_sampling_algorithm.html

    Runs in O(lambda)
    """
    k = 1
    t = np.exp(-lmbda) / (1 - np.exp(-lmbda)) * lmbda
    s = t
    u = np.random.random()

    while s < u:
        k += 1
        t *= lmbda / k
        s += t
    return k


def sample_distribution_tuple(dist_list):
    """Sample a distribution tuple from a list of distribution dictionaries (or a single distribution)"""
    # If there is just a single element, no need to sample
    if not dist_list:
        return None

    if type(dist_list) == dict:
        return get_distribution_tuple(dist_list)

    weights = [dist.get('weight', 1) for dist in dist_list]
    dist = random.choices(dist_list, weights=weights)[0]
    return get_distribution_tuple(dist)


def run(fg_folder, bg_folder, scaper_spec_path, n_soundscapes, out_folder, exp_label, random_state=123):
    # create output directory
    os.makedirs(out_folder, exist_ok=True)

    # load Scaper specifications
    with open(scaper_spec_path, 'r') as f:
        scaper_spec = json.load(f)

    # create a scaper that will be used below
    sc = scaper.Scaper(scaper_spec['duration'],
                       fg_folder,
                       bg_folder,
                       protected_labels=scaper_spec.get('protected_labels', []),
                       random_state=random_state)

    sc.sr = scaper_spec['sample_rate']
    sc.ref_db = scaper_spec['ref_db']

    # save generation params
    gen_params = {
        'fg_folder': fg_folder,
        'bg_folder': bg_folder,
        'scaper_spec_path': scaper_spec_path,
        'n_soundscapes': n_soundscapes,
        'out_folder': out_folder,
        'exp_label': exp_label,
        'random_state': random_state,
        'fg_labels': sc.fg_labels,
        'bg_labels': sc.bg_labels
    }

    gen_params_path = os.path.join(out_folder, "soundscape_gen_params_{}.json".format(exp_label))
    with open(gen_params_path, 'w') as f:
        json.dump(gen_params, f)


    print('Generating soundscapes.')
    for n in tqdm(range(n_soundscapes), total=n_soundscapes):
        # reset the event specifications for foreground and background at the
        # beginning of each loop to clear all previously added events
        sc.reset_bg_event_spec()
        sc.reset_fg_event_spec()

        # sample distributions for background parameters
        bg_label = sample_distribution_tuple(scaper_spec['background_label'])

        if bg_label:
            bg_source_file = sample_distribution_tuple(scaper_spec['background_source_file'])
            bg_source_time = sample_distribution_tuple(scaper_spec['background_source_time'])

            # add background
            sc.add_background(label=bg_label,
                              source_file=bg_source_file,
                              source_time=bg_source_time)

        # add random number of foreground events
        if scaper_spec['num_events'] is None:
            n_events = 0
        elif scaper_spec['num_events']['dist'] == 'uniform':
            n_events = np.random.randint(scaper_spec['num_events']['min'],
                                         scaper_spec['num_events']['max'] + 1)
        elif scaper_spec['num_events']['dist'] == 'ztpoisson':
            n_events = sample_ztpoisson(scaper_spec['num_events']['rate'])
        else:
            raise ValueError('Invalid distribution for number of events: {}'.format(scaper_spec['num_events']['dist']))

        for _ in range(n_events):
            # sample distributions for event parameters
            fg_label = sample_distribution_tuple(scaper_spec['event_label'])
            fg_source_file = sample_distribution_tuple(scaper_spec['event_source_file'])
            fg_source_time = sample_distribution_tuple(scaper_spec['event_source_time'])
            event_time = sample_distribution_tuple(scaper_spec['event_time'])
            event_duration = sample_distribution_tuple(scaper_spec['event_duration'])
            snr = sample_distribution_tuple(scaper_spec['snr'])
            pitch_shift = sample_distribution_tuple(scaper_spec['pitch_shift'])
            time_stretch = sample_distribution_tuple(scaper_spec['time_stretch'])

            sc.add_event(label=fg_label,
                         source_file=fg_source_file,
                         source_time=fg_source_time,
                         event_time=event_time,
                         event_duration=event_duration,
                         snr=snr,
                         pitch_shift=pitch_shift,
                         time_stretch=time_stretch)

        # generate
        audiofile = os.path.join(out_folder, "soundscape_{}_{:d}.wav".format(exp_label, n))
        jamsfile = os.path.join(out_folder, "soundscape_{}_{:d}.jams".format(exp_label, n))

        # TODO: If we want we can parameterize the other arguments of sc.generate at some point...
        sc.generate(audiofile, jamsfile,
                    allow_repeated_label=scaper_spec['allow_repeated_label'],
                    allow_repeated_source=scaper_spec['allow_repeated_source'],
                    reverb=scaper_spec['reverb'],
                    disable_sox_warnings=True,
                    no_audio=False,
                    save_isolated_events=True)


def parse_arguments(args):
    parser = ArgumentParser(sys.argv[0],
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('fg_folder', type=str,
                        help='Path to foreground audio directory')
    parser.add_argument('bg_folder', type=str,
                        help='Path to background audio directory')
    parser.add_argument('scaper_spec_path', type=str,
                        help='Path to Scaper JSON specification file.')
    parser.add_argument('n_soundscapes', type=positive_int,
                        help='Path to background audio directory')
    parser.add_argument('exp_label', type=str,
                        help='Label to use for this set of generated audio clips.')
    parser.add_argument('out_folder', type=str,
                        help='Output directory where audio files will be saved.')
    parser.add_argument('--random-state', type=int, default=123,
                        help='Random seed to set for reproducability.')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    run(fg_folder=args.fg_folder,
        bg_folder=args.bg_folder,
        scaper_spec_path=args.scaper_spec_path,
        n_soundscapes=args.n_soundscapes,
        exp_label=args.exp_label,
        out_folder=args.out_folder,
        random_state=args.random_state)

