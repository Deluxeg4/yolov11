# read_bottle_count.py

import os

def read_bottle_count(filename='count.txt'):
    """
    Reads the final bottle count from a specified text file.

    Args:
        filename (str): The name of the file containing the bottle count.
                        Defaults to 'count.txt'.

    Returns:
        int or None: The final bottle count as an integer if found, otherwise None.
    """
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found. "
              "Please make sure the YOLO detection script has been run and generated the file.")
        return None

    try:
        with open(filename, 'r') as f:
            content = f.read()
            # The format is expected to be "Final bottle count: X"
            if "Final bottle count:" in content:
                # Extract the number after "Final bottle count: "
                count_str = content.split("Final bottle count:")[-1].strip()
                try:
                    count = int(count_str)
                    return count
                except ValueError:
                    print(f"Error: Could not parse bottle count from file '{filename}'. Content: '{content}'")
                    return None
            else:
                print(f"Error: Expected 'Final bottle count:' in file '{filename}', but found: '{content}'")
                return None
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        return None

if __name__ == "__main__":
    bottle_count = read_bottle_count()

    if bottle_count is not None:
        print(f"The final bottle count is: {bottle_count}")
    else:
        print("Failed to retrieve the final bottle count.")

