using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFrame
{
    public static class FunctionProvider
    {
        public static Func<double, double> GetFunction(Functions function)
        {
            switch (function)
            {
                case Functions.Sigmoid:
                    return FunctionProvider.Sigmoid;
                case Functions.ReLu:
                    return FunctionProvider.ReLU;
                case Functions.LeakyReLu:
                    return FunctionProvider.Leaky_ReLU;
                case Functions.None:
                    return x => { return x; };
                default:
                    throw new ArgumentException("Functions is not implemented");
            }
        }

        public static Func<double, double> GetFunctionDerivative(Functions function)
        {
            switch (function)
            {
                case Functions.Sigmoid:
                    return FunctionProvider.DerivedSigmoid;
                default:
                    throw new ArgumentException("Functions is not implemented");
            }
        }

        internal static Vector<double> SoftMax(Vector<double> input)
        {
            var expInput = input.PointwiseExp();
            double eSum = expInput.Sum();

            return expInput.Divide(eSum);
        }

        internal static double DerivedSoftMax(Vector<double> input, int indexOfOutputElement, int indexOfRespectedElement)
        {
            Vector<double> softmaxValues = SoftMax(input);
            var outputElementSoftmax = softmaxValues[indexOfOutputElement];

            if (indexOfOutputElement == indexOfRespectedElement)
            {
                return outputElementSoftmax * (1 - outputElementSoftmax);
            }
            else
            {
                return (outputElementSoftmax * -1) * softmaxValues[indexOfRespectedElement];
            }
        }

        #region Activation Functions

        internal static double Sigmoid(double x)
        {
            return 1 / (1 + (double)Math.Pow(Math.E, x * -1d));
        }

        internal static double ReLU(double x)
        {
            if (x > 0)
            {
                return x;
            }
            return 0f;
        }

        internal static double Leaky_ReLU(double x)
        {
            if (x > 0)
            {
                return FunctionProvider.ReLU(x);
            }

            return 0.01f * x;
        }

        #endregion

        #region Activation Function Derivatives

        internal static double DerivedSigmoid(double x)
        {
            var sigmoidX = FunctionProvider.Sigmoid(x);

            return sigmoidX * (1 - sigmoidX);
        }

        #endregion


        #region Loss Functions

        internal static double MeanSquaredError(Vector<double> prediction, Vector<double> target)
        {
            Vector<double> resultVector = CreateVector.Dense<double>(prediction.Count);
            prediction.Subtract(target, resultVector);

            resultVector = resultVector.PointwisePower(2f);
            var sum = resultVector.Sum();

            return sum / (double)prediction.Count;
        }

        internal static double CrossEntropyLossSum(double[] prediction, double[] target)
        {
            double loss = 0;
            // log loss for class x is => -1*[target prediction] * ln([actual prediction])
            for (int i = 0; i < prediction.Length; i++)
            {
                loss += (double)(-1 * target[i] * Math.Log(prediction[i]));
            }

            return loss;
        }

        internal static double CrossEntropyLoss(double prediction, double target)
        {
            // log loss for class x is => -1*[target prediction] * ln([actual prediction])
            return (double)((-1 * target) * Math.Log(prediction));
        }

        internal static double CategorialCrossEntropyLoss(double prediction)
        {
            // log loss for class x is => -1*[target prediction] * ln([actual prediction])
            return (double)((-1) * Math.Log(prediction));
        }

        #endregion

        #region Loss Functions Derivatives

        internal static double DerivedCategorialCrossEntropyLoss(double prediction, double target)
        {
            // -(t/p)
            return (double)((-1*target)/prediction);
        }

        #endregion
    }
}
