#include "calculator.h"
#include <cmath>

int Calculator::add(int a, int b) { return a + b; }
int Calculator::subtract(int a, int b) { return a - b; }
int Calculator::multiply(int a, int b) { return a * b; }

double Calculator::divide(int a, int b)
{
    if (b == 0)
        return std::nan("");
    return static_cast<double>(a) / static_cast<double>(b);
}