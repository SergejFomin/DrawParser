using System;
using System.Diagnostics.CodeAnalysis;

namespace WpfAppdrawtest
{
    public class ClassificationViewModel
    {
        public ClassificationViewModel(int classification, double probability)
        {
            this.Classification = classification;
            this.Probability = probability;
        }

        public int Classification { get; }

        public double Probability { get; }
    }
}