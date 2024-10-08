#helper functionss

import argparse

def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Process the file content here
        print(content)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read and process a file.")
    
    # Add a positional argument for the file path
    parser.add_argument("file", help="Path to the file to be processed")
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call the file processing function with the file path
    process_file(args.file)