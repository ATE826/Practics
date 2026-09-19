using System;
using System.Collections.Generic;
using System.IO;

namespace RefactoringExample
{
    public class Employee
    {
        public string Id { get; set; }
        public int HoursWorked { get; set; }
        public double Rate { get; set; }
        public string Grade { get; set; }
        public bool HasInsurance { get; set; }
        public bool IsTaxExempt { get; set; }
        public string Department { get; set; }
        public int YearsOfExperience { get; set; }
    }

    public static class DateFormatter
    {
        public static string FormatToday()
        {
            return DateTime.Today.ToString("yyyy-MM-dd");
        }
    }

    // Расчётная логика
    public class SalaryCalculator{
        // Магические числа 0.2 / 0.1 / 0.05 заменены на именованную таблицу
        // (Replace Conditional with Lookup Table + Replace Magic Number with Symbolic Constant).
        private static readonly Dictionary<string, double> AllowanceRates = new Dictionary<string, double>
        {
            { "A", 0.2 },
            { "B", 0.1 },
            { "C", 0.05 }
        };

        private const double InsuranceDeductionRate = 0.05;   // ставка удержания за страховку — 5%
        private const double TaxDeductionRate = 0.1;          // ставка удержания налога — 10%

        // Базовый оклад
        public double CalculateBaseSalary(Employee emp)
        {
            return emp.HoursWorked * emp.Rate;
        }

        // Считает размер надбавки в зависимости от грейда сотрудника (A/B/C).
        public double CalculateAllowance(double baseSalary, string grade)
        {
            return AllowanceRates.TryGetValue(grade, out var rate) ? baseSalary * rate : 0;
        }

        // Считает сумму всех удержаний (страховка + налог).
            public double CalculateDeductions(double baseSalary, Employee emp)
        {
            double deductions = 0;

            if (emp.HasInsurance)
            {
                deductions += baseSalary * InsuranceDeductionRate;
            }

            if (!emp.IsTaxExempt)
            {
                deductions += baseSalary * TaxDeductionRate;
            }

            return deductions;
        }

        // Считает итоговую зарплату сотрудника
        public double CalculateNetSalary(Employee emp)
        {
            double baseSalary = CalculateBaseSalary(emp);
            double allowance = CalculateAllowance(baseSalary, emp.Grade);
            double deductions = CalculateDeductions(baseSalary, emp);

            return baseSalary + allowance - deductions;
        }
    }

    // Определение грейда по стажу работы
    public class GradeCalculator
    {
        private static readonly Dictionary<string, (int Senior, int Mid)> DepartmentThresholds =
            new Dictionary<string, (int Senior, int Mid)>
            {
                { "IT", (10, 5) },
                { "HR", (8, 4) }
            };

        public string GetEmployeeGrade(Employee emp)
        {
            if (!DepartmentThresholds.TryGetValue(emp.Department, out var thresholds))
            {
                return "C";
            }

            if (emp.YearsOfExperience > thresholds.Senior) return "A";
            if (emp.YearsOfExperience > thresholds.Mid) return "B";
            return "C";
        }
    }

    // Вся консольная диагностика
    public class ConsoleLogger
    {
        public void LogSalaryCalculationStart(string employeeId)
        {
            Console.WriteLine($"Starting salary calculation for employee {employeeId}");
        }

        public void LogSalaryCalculated(string employeeId, string date)
        {
            Console.WriteLine($"Salary calculated for employee {employeeId} on {date}");
        }

        public void LogReportGenerated(string date)
        {
            Console.WriteLine($"Report generated on {date}");
        }
    }

        // Запись в файл
    public class SalaryReportWriter
    {
        private const string ReportFileName = "salary_report.txt";

        public void AppendSalaryRecord(string employeeId, double netSalary, string date)
        {
            File.AppendAllText(ReportFileName, $"Employee {employeeId}: {netSalary} on {date}\n");
        }
    }

        // Связывание всего функционала
    public class SalaryService
    {
        private readonly SalaryCalculator _calculator = new SalaryCalculator();
        private readonly ConsoleLogger _logger = new ConsoleLogger();
        private readonly SalaryReportWriter _reportWriter = new SalaryReportWriter();

        public double CalculateSalary(Employee emp)
        {
            _logger.LogSalaryCalculationStart(emp.Id);

            double netSalary = _calculator.CalculateNetSalary(emp);
            string date = DateFormatter.FormatToday();

            _logger.LogSalaryCalculated(emp.Id, date);
            _reportWriter.AppendSalaryRecord(emp.Id, netSalary, date);

            return netSalary;
        }
    }

    public class Program
    {
        public static void GenerateReport()
        {
            var logger = new ConsoleLogger();
            logger.LogReportGenerated(DateFormatter.FormatToday());
        }

        static void Main(string[] args)
        {
            var employee = new Employee
            {
                Id = "emp123",
                HoursWorked = 160,
                Rate = 25.0,
                Grade = "B",
                HasInsurance = true,
                IsTaxExempt = false,
                Department = "IT",
                YearsOfExperience = 7
            };

            var salaryService = new SalaryService();
            double salary = salaryService.CalculateSalary(employee);
            Console.WriteLine($"Calculated salary: {salary}");

            var gradeCalculator = new GradeCalculator();
            string grade = gradeCalculator.GetEmployeeGrade(employee);
            Console.WriteLine($"Employee grade: {grade}");

            GenerateReport();
        }
        
    }
}