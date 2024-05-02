"""
This script reads a JSON or CSV file and reduces the number of elements to a specified count.
"""

import sys
import json
import csv
import os


def reduce_file(element_count, input_file_path, output_file_name):
    """
    Reduce the number of elements in a JSON or CSV file to the specified count.
    :param element_count:       Number of elements to keep
    :param input_file_path:     Path to the input file
    :param output_file_name:    Name of the output file
    :return:                    None
    """
    try:
        file_extension = os.path.splitext(input_file_path)[1]

        # Process JSON files
        if file_extension == '.json':
            with open(input_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if not isinstance(data, list):
                print("JSON is not a list. Please provide a JSON file with a list of elements.")
                return

            reduced_data = data[:element_count]

            with open(output_file_name, 'w', encoding='utf-8') as file:
                json.dump(reduced_data, file, indent=4)

        # Process CSV files
        elif file_extension == '.csv':
            with open(input_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='|')
                data = list(reader)

            reduced_data = data[:element_count + 1]  # Include header row

            with open(output_file_name, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(reduced_data)

        else:
            print(f"Unsupported file format: {file_extension}")
            return

        print(f"{element_count} elements have been written to {output_file_name}")

    except FileNotFoundError:
        print(f"The file {input_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file {input_file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python reduce_file.py [element_count] [input_file_path] [output_file_name]")
    else:
        _, element_count, input_file_path, output_file_name = sys.argv
        reduce_file(int(element_count), input_file_path, output_file_name)
