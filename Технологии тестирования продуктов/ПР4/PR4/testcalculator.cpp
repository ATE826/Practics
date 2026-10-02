#include "testcalculator.h"

void TestCalculator::initTestCase() { qDebug("Начало тестирования класса Calculator"); }
void TestCalculator::cleanupTestCase() { qDebug("Тестирование класса Calculator завершено"); }
void TestCalculator::init() { calc = new Calculator(); }
void TestCalculator::cleanup() { delete calc; calc = nullptr; }

void TestCalculator::testAddition()
{
    QCOMPARE(calc->add(2, 3), 5);
    QCOMPARE(calc->add(-1, 1), 0);
    QCOMPARE(calc->add(0, 0), 0);
    QCOMPARE(calc->add(-5, -3), -8);
}

void TestCalculator::testSubtraction()
{
    QCOMPARE(calc->subtract(5, 3), 2);
    QCOMPARE(calc->subtract(3, 5), -2);
    QCOMPARE(calc->subtract(0, 0), 0);
    QCOMPARE(calc->subtract(-4, -4), 0);
    QCOMPARE(calc->subtract(-2, 3), -5);
}

void TestCalculator::testMultiplication()
{
    QCOMPARE(calc->multiply(2, 3), 6);
    QCOMPARE(calc->multiply(-2, 3), -6);
    QCOMPARE(calc->multiply(-2, -3), 6);
    QCOMPARE(calc->multiply(0, 5), 0);
}

void TestCalculator::testDivision()
{
    QCOMPARE(calc->divide(10, 2), 5.0);
    QCOMPARE(calc->divide(-9, 3), -3.0);
    QCOMPARE(calc->divide(0, 5), 0.0);
    QVERIFY2(qIsNaN(calc->divide(10, 0)), "Деление на ноль должно возвращать NaN");
    QVERIFY2(qIsNaN(calc->divide(0, 0)), "0 / 0 тоже должно возвращать NaN");
}

QTEST_MAIN(TestCalculator)