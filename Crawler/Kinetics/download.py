#!/usr/bin/env python

import argparse
import glob
import json
import os
import shutil
import subprocess
import uuid
from collections import OrderedDict

from joblib import delayed
from joblib import Parallel
import pandas as pd

import sys
import random

def create_video_folders(dataset, output_dir, tmp_dir):
    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = os.path.join(output_dir, 'test')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        # I should return a dict but ...
        return this_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename

def download_clip(video_identifier, output_filename,
                  start_time, end_time,
                  tmp_dir='/tmp/kinetics',
                  num_attempts=5,
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'
    status = False
    proxy=get_random_proxy()
    # Construct command line for getting the direct video link.
    command = ['youtube-dl',
               '-f', '18', # 640x360 h264 encoded video
               '--proxy', proxy,
               '--get-url',
               '"%s"' % (url_base + video_identifier)]
    command = ' '.join(command)
    direct_download_url = None
    attempts = 0
    while True:
         try:
            direct_download_url = subprocess.check_output(command,
                                                          shell=True,
                                                          stderr=subprocess.STDOUT)
            direct_download_url = direct_download_url.strip().decode('utf-8')
         except subprocess.CalledProcessError as err:
            if "429" in err.output:
               remove_proxy_from_list(proxy)
            print('{} - {}, proxy {}'.format(video_identifier, err, proxy), file=sys.stdout)
            attempts += 1
            if attempts == num_attempts:
                return status, str(err.output)
            else:
                continue
         break
    # Construct command to trim the videos (ffmpeg required).
    command = ['http_proxy={}'.format(proxy),
               'https_proxy={}'.format(proxy),
               'HTTP_proxy={}'.format(proxy),
               'HTTPS_proxy={}'.format(proxy),
               'ffmpeg',
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-i', "'%s'" % direct_download_url,
               '-c:v', 'libx264', '-preset', 'ultrafast',
               '-c:a', 'aac',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        if "429" in err.output:
            remove_proxy_from_list(proxy)
        print('{} - {}'.format(video_identifier, err), file=sys.stdout)
        return status, str(err.output)

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    print('{} - downloaded - proxy: {}'.format(video_identifier,proxy ), file=sys.stdout)
    return status, 'Downloaded'

def get_random_proxy():
    try:
        f = open("proxies.txt", "r")
    except:
        print('failed reading proxies.txt file')
        exit(1)
    text = f.read()
    f.close()

    proxies = text.split("\n")
    proxies.remove('')

    proxy = random.choice(proxies)
    if not proxy:
        print('no proxy value')
        exit(1)
    else:
        print('Using {} proxy'.format(proxy), file=sys.stdout)
    return proxy

def remove_proxy_from_list(proxy):
    with open("proxies.txt", "r") as f:
        lines = f.readlines()
    with open("proxies.txt", "w") as f:
        for line in lines:
            if line.strip("\n") != proxy:
                f.write(line)
    disable_squid_on_srv(proxy)


def disable_squid_on_srv(proxy):
    import subprocess
    ip= proxy.strip("https://").split(":")[0]
    ssh_command = "ssh {}".format(ip)
    stop = ["ssh","-o", "StrictHostKeychecking=no", ip, "sudo", "systemctl" ,"stop", "squid"]
    disable = ["ssh", "-o", "StrictHostKeychecking=no", ip, "sudo", "systemctl" ,"disable", "squid"]
    subprocess.run(stop)
    subprocess.run(disable)


def download_clip_wrapper(row,
                          label_to_dir,
                          trim_format,
                          tmp_dir,
                          csv_status_file):
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(row, label_to_dir,
                                               trim_format)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists'])
        return status

    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)

    error_429_message = "HTTP Error 429: Too Many Requests"
    if csv_status_file is not None and error_429_message not in log:
        with open(csv_status_file, 'a') as f:
            f.write('\n{}, {}'.format(
                row['video-id'], str(log).replace(',', '.')
            ))
    status = tuple([clip_id, downloaded, log])
    return status


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


def main(input_csv, output_dir,
         trim_format='%06d', num_jobs=24, tmp_dir='/tmp/kinetics',
         drop_duplicates=False, csv_status_file=None):

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)
    # if os.path.isfile(drop_duplicates):
    #     print('Attempt to remove duplicates')
    #     old_dataset = parse_kinetics_annotations(drop_duplicates,
    #                                              ignore_is_cc=True)
    #     df = pd.concat([dataset, old_dataset], axis=0, ignore_index=True)
    #     df.drop_duplicates(inplace=True, keep=False)
    #     print(dataset.shape, old_dataset.shape)
    #     dataset = df
    #     print(dataset.shape)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)

    if csv_status_file is not None:
        if not os.path.exists(csv_status_file):
            with open(csv_status_file, 'a') as f:
                f.write('video_identifier, status')
        status_df = pd.read_csv(csv_status_file)
        index_values = dataset[dataset['video-id'].isin(
            status_df.video_identifier.unique())].index
        dataset = dataset.drop(index_values).reset_index(drop=True)

    # Download all clips.
    if num_jobs == 1:
        status_lst = []
        for i, row in dataset.iterrows():
            status_lst.append(download_clip_wrapper(row, label_to_dir,
                                                    trim_format, tmp_dir,
                                                    csv_status_file))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(delayed(download_clip_wrapper)(
            row, label_to_dir,
            trim_format, tmp_dir,
            csv_status_file) for i, row in dataset.iterrows())

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)

    # Save download report.
    with open('download_report.json', 'w') as fobj:
        fobj.write(json.dumps(status_lst))


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('-f', '--trim-format', type=str, default='%06d',
                   help=('This will be the format for the '
                         'filename of trimmed videos: '
                         'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    p.add_argument('--drop-duplicates', type=str, default='non-existent',
                   help='Unavailable at the moment')
                   # help='CSV file of the previous version of Kinetics.')
    p.add_argument('-c', '--csv-status-file', type=str,
                   help='CSV file containing files that have been already'
                        'processed')
    main(**vars(p.parse_args()))
