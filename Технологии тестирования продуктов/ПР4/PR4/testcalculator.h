#ifndef TESTCALCULATOR_H
#define TESTCALCULATOR_H

#include <QtTest>
#include "calculator.h"

class TestCalculator : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

    void testAddition();
    void testSubtraction();
    void testMultiplication();
    void testDivision();

private:
    Calculator *calc = nullptr;
};

#endif // TESTCALCULATOR_H