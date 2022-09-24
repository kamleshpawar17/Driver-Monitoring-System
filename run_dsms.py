import argparse
import yaml

from dsms_core import calibration, run

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--showvideo", type=int, default=True)
ap.add_argument("-m", "--multiproc", type=int, default=True)
ap.add_argument("-np", "--numproc", type=int, default=2)
ap.add_argument("-c", "--calib", type=int, default=True)
ap.add_argument("-sav", "--savevideo", type=int, default=False)
args = vars(ap.parse_args())


if __name__ == '__main__':
    # ----- read config refer to config.yaml for description of parameters---- #
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # --- Command line arguments --- #
    config['MULTIPROC_FLAG'] = args['multiproc']
    config['NUM_PROC'] = args['numproc']
    config['SHOW_IMG'] = args['showvideo']
    config['RUN_CALIB'] = args['calib']
    config['SAVE_VID'] = args['savevideo']

    if config['RUN_CALIB']:
        calibration(config)

    run(config)
