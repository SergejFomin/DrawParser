using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DrawParser.Core
{
    public sealed class TrainingData : IComparable<TrainingData>
    {
        private double[] data;

        public TrainingData(double[] data, int expected)
        {
            this.data = data ?? throw new ArgumentNullException(nameof(data));
            this.Expected = expected;
        }

        public double[] Data
        {
            get
            {
                // todo: maybe return copy
                return data; 
            }
        }

        public int Expected { get; private set; }

        public int CompareTo(TrainingData? other)
        {
            _ = other ?? throw new ArgumentNullException(nameof(other));

            if (this.Expected < other.Expected)
            {
                // preceding
                return -1;
            }
            else if (this.Expected > other.Expected)
            {
                // following
                return 1;
            }

            return 0;
        }
    }
}
