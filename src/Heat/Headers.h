//
// Created by giley on 11/12/2025.
//

#ifndef HEADERS_H
#define HEADERS_H
#include <string>

using real = float;
real *data();

void SaveToCSV(real *T, const std::string& filename);
void saveToBin(real *T, const std::string &filename);
#endif //HEADERS_H
