// вариант 3: поиск пересечения двух массивов

using System;
using System.Collections.Generic;
using System.Diagnostics;

class Program
{
    static List<int> FindIntersection(int[] arr1, int[] arr2)
    {
        HashSet<int> set = [..arr1];
        HashSet<int> foundIntersections = new HashSet<int>();

        foreach(int value in arr2)
        {
            if (set.Contains(value))
            {
                foundIntersections.Add(value);
            }
        }
        return [..foundIntersections];
    }

    static void Main()
    {
        int[] arr1 = { 1, 2, 3, 4, 5};
        int[] arr2 = { 3, 4, 5, 6, 7 };
        Stopwatch sw = Stopwatch.StartNew();
        List<int> foundIntersections = FindIntersection(arr1, arr2);
        sw.Stop();
        Console.WriteLine("Пересечение: " + string.Join(", ", foundIntersections));
        Console.WriteLine("Время: " + sw.ElapsedTicks + " тиков");
    }
}