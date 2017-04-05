# -*- coding: utf-8 -*-

# Data Cleansing
# Change the json file into csv file

import argparse
import csv
import json

def read_and_write_json(json_file_path, csv_file_path):
    """Read in the json dataset file and write it out to a csv file"""
    with open(csv_file_path, "wb+") as fout:
        with open(json_file_path) as fin:
            csv_file = csv.writer(fout)
            first_item = True
            for line in fin:
                data = json.loads(line)
                if first_item:
                    column_names = data.keys()
                    csv_file.writerow(column_names)
                    csv_file.writerow(get_row(data ,column_names))
                    first_item = False
                else:
                    column_values = data.values()
                    csv_file.writerow(get_row(data ,column_names))


def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = line_contents[column_name]
        if isinstance(line_value, unicode):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    parser = argparse.ArgumentParser(
            description='Convert Yelp Dataset Challenge data from JSON format to CSV.',
            )

    parser.add_argument(
            'json_file',
            type=str,
            help='The json file to convert.',
            )

    args = parser.parse_args()

    json_file = args.json_file
    csv_file = '{0}.csv'.format(json_file.split('.json')[0])
    read_and_write_json(json_file, csv_file)
