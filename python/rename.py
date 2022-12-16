#!/usr/bin/env python3


import pathlib
import sys


def scan_and_replace(sys_argv):
    root = pathlib.Path(sys_argv[1])
    n_steps_per_frames = int(sys_argv[2])
    old_str = sys_argv[3]
    new_str = sys_argv[4]
    paths = list(root.glob('./' + old_str + '[0-9]*'))
    for path in paths:
        old_name = str(path)
        old_str_pos = old_name.find(old_str)
        prefix = old_name[0:old_str_pos]
        suffix = old_name[old_str_pos + len(old_str):]
        dot_pos = suffix.find('.')
        if (dot_pos == -1):
            dot_pos = len(suffix)
        old_short_name = old_str + suffix[0:dot_pos]
        new_short_name = new_str + str(int(suffix[0:dot_pos]) // n_steps_per_frames)
        suffix = suffix[dot_pos:]
        new_name = prefix + new_short_name + suffix
        if (len(suffix)):
            print(old_short_name, '->', new_short_name)
            with path.open() as old_file:
                with open(new_name, 'w') as new_file:
                    for line in old_file:
                        new_file.write(line.replace(old_short_name, new_short_name))
            pathlib.Path(new_name).replace(old_name)
        pathlib.Path(old_name).replace(new_name)
    pathlib.Path(root.name + '/CHECK_CGNS.txt').touch()

if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print('Usage:')
        print('    python3 rename.py <root> <n_steps_per_frames> <old_str> <new_str>')
    else:
        scan_and_replace(sys.argv)
