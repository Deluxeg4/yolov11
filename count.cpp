// read_bottle_count.cpp

#include <iostream> // For console input/output (includes std::cout, std::endl)
#include <fstream>  // For file stream operations
#include <string>   // For string manipulation
#include <filesystem> // For checking file existence (C++17 or later)

namespace fs = std::filesystem; // Use fs namespace for std::filesystem

// Function to read the final bottle count from a specified text file
int readBottleCount(const std::string& filename = "count.txt") {
    // Check if the file exists
    if (!fs::exists(filename)) {
        std::cerr << "Error: The file '" << filename << "' was not found.\n"
                  << "Please make sure the YOLO detection script has been run and generated this file.\n";
        return -1; // Return -1 to indicate an error or file not found
    }

    std::ifstream infile(filename); // Open the file for reading
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open the file '" << filename << "' for reading.\n";
        return -1; // Return -1 to indicate an error
    }

    std::string line;
    int bottle_count = -1; // Default to -1 indicating not found or error

    // Read each line from the file
    while (std::getline(infile, line)) {
        // Look for the specific string "Final bottle count: "
        std::string search_str = "Final bottle count: ";
        size_t pos = line.find(search_str); // Find the position of the search string

        if (pos != std::string::npos) { // If the string is found
            // Extract the substring after the search string
            std::string count_str = line.substr(pos + search_str.length());
            try {
                // Convert the extracted string to an integer
                bottle_count = std::stoi(count_str);
                break; // Found the count, no need to read further
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Error: Could not parse bottle count from line: '" << line << "'\n"
                          << "Details: " << ia.what() << "\n";
                bottle_count = -1; // Indicate parsing error
            } catch (const std::out_of_range& oor) {
                std::cerr << "Error: Bottle count out of range from line: '" << line << "'\n"
                          << "Details: " << oor.what() << "\n";
                bottle_count = -1; // Indicate parsing error
            }
        }
    }

    infile.close(); // Close the file
    return bottle_count; // Return the found count or -1 if an error occurred/not found
}

int main() {
    // Call the function to read the bottle count
    int count = readBottleCount();

    // Calculate points (assuming 0 points if count is -1 due to error)
    int points = (count != -1) ? (count * 2) : 0; // If count is -1, set points to 0, otherwise calculate.

    // Display the result (points)
    std::cout << "Calculated points: " << points << std::endl;

    // You can also choose to display the raw count and an error message if count is -1
    if (count == -1) {
        std::cerr << "Note: Failed to retrieve a valid bottle count, points might be 0.\n";
    }

    return 0;
}
