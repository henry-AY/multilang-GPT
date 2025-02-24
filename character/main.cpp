#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <map>

/**
 * run from root project dir
 * 
 * g++ character/main.cpp -o a.out
 * ./a.out <Language>/<Data Set Name>.txt
*/

int countChar(const std::string& filePath, const std::string& outputFile) {
    std::ifstream input(filePath);
    std::ofstream output(outputFile);
    std::map<char, int> char_freq;
    
    if(!input.is_open()) {
        std::cerr << "Error opening " << filePath << std::endl;
        input.close();
        return 1;
    }

    if(!output.is_open()) {
        std::cerr << "Error writing to " << outputFile << std::endl;
        input.close();
        return 1;
    }

    char ch;
    while(input.get(ch)) {
        char_freq[ch]++;
    }
    input.close();

    output << "Character,Count\n";
    for(const auto& ch_map : char_freq) {
        output << ch_map.first << "," << ch_map.second << "\n";
    }
    output.close();
    return 0;
}  

int main(int argc, char* argv[]) {
    if(argc != 2) { 
        std::cerr << "Only submit one file" << std::endl;
        return 1; 
    }
    std::string outputFile = "character/output/output.csv";

    return countChar(argv[1], outputFile);
}