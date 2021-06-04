import os
import pandas as pd
from collections import OrderedDict
from glob import glob
import argparse

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # This line is critical to keep consistency of the class definition
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = OrderedDict([
            ('youtube_id', 'video-id'),
            ('time_start', 'start-time'),
            ('time_end', 'end-time'),
            ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df

def get_split_info(data_path,input_csv, class_to_idx, trim_format='%06d'):
    df = parse_kinetics_annotations(input_csv)
    count = 0
    split_info = []
    for i, row in df.iterrows():
        label_name = row['label-name']
        basename = '%s_%s_%s' % (row['video-id'],
                                    trim_format % row['start-time'],
                                    trim_format % row['end-time'])
        
        dirname = os.path.join(data_path,label_name,basename)
        if not os.path.exists(dirname):
            count += 1
            continue
        
        frames = glob("{}/*.jpg".format(dirname))
        if len(frames) < 1:
            print(i,dirname," has no frame found!")
            continue
        split_info.append([os.path.join(label_name, basename),len(frames),class_to_idx[label_name]])
    print("Missing {} files in {}".format(count, input_csv))
    return split_info

def write_split_files(save_path, split_id, phase, split_info):
    # split_id = 1
    # save_path = '.datasets/settings/kinetics100/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    split_file = os.path.join(save_path,"{}_rgb_split{}.txt".format(phase,split_id))

    with open(split_file, 'w') as f:
        for item in split_info:
            item = [str(_) for _ in item]
            f.write("%s\n" % ",".join(item))

def main(args):
    frame_path = args.frames_path
    setting_path = args.settings
    split_id = args.split_id
    classes, class_to_idx = find_classes(frame_path)
    input_csv_list = glob('{}/*.csv'.format(setting_path))

    for input_csv in input_csv_list:
        fn = os.path.basename(input_csv)
        if 'train' in fn:
            phase = 'train'
        elif 'val' in fn:
            phase = 'val'
        else:
            phase = 'test'
        split_info = get_split_info(frame_path, input_csv, class_to_idx)
        write_split_files(setting_path, split_id, phase, split_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate split files for kinetics 100')
    parser.add_argument('--settings', metavar='DIR', default='./datasets/settings/kinetics100',
                        help='path to dataset setting files')
    parser.add_argument('--frames-path', metavar='DIR', default='./datasets/kinetcis100_frames',
                        help='path to dataset files')  
    parser.add_argument('--split-id', default=1, type=int, metavar='S',
                    help='id of the split files, e.g. train{split-id}.txt')  
    args = parser.parse_args()
    main(args)