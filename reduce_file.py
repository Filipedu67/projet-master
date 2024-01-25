import sys
import json


def reduce_file(element_count, input_file_path, output_file_name):
    try:
        # Read the input JSON file
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Check if the data is a list
        if not isinstance(data, list):
            print("JSON is not a list. Please provide a JSON file with a list of elements.")
            return

        # Reduce the data to the first N elements
        reduced_data = data[:element_count]

        # Write the reduced data to the new file
        with open(output_file_name, 'w', encoding='utf-8') as file:
            json.dump(reduced_data, file, indent=4)

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
