using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkFrame
{
    public sealed class NeuralNetwork
    {
        internal NeuralNetworkConfiguration configuration;
        internal InternalNeuralNetwork neuralNetwork;

        public NeuralNetwork(NeuralNetworkConfiguration neuralNetworkConfiguration)
        {
            this.configuration = neuralNetworkConfiguration ?? throw new ArgumentNullException(nameof(neuralNetworkConfiguration));
            this.neuralNetwork = new InternalNeuralNetwork(configuration, true);
        }

        public double[] CalculateOutput(double[] input)
        {
            var output = this.neuralNetwork.CalculateOutput(input);

            if (configuration.SoftMaxResult)
            {
                return FunctionProvider.SoftMax(output).ToArray();
            }

            return output.ToArray();
        }

        public NeuralNetworkTrainer GetNeuralNetworkTrainer(ITraingingDataProvider traingingDataProvider)
        {
            _ = traingingDataProvider ?? throw new ArgumentNullException(nameof(traingingDataProvider));

            return new NeuralNetworkTrainer(this, traingingDataProvider);
        }

        public void SaveWeightsAndBias(string filePath)
        {
            var weights = this.neuralNetwork.GetFlatWeights();
            var byteArray = new byte[weights.Length * sizeof(double)];
            Buffer.BlockCopy(weights, 0, byteArray, 0, byteArray.Length);
            using (var stream = File.OpenWrite(filePath))
            {
                using (MemoryStream weightsStream = new MemoryStream(byteArray))
                {
                    weightsStream.WriteTo(stream);
                }
            }
        }

        public void LoadWeightsAndBias(string filePath)
        {
            if (!File.Exists(filePath))
            {
                return;
            }

            byte[] byteArray;
            using(var stream = File.OpenRead(filePath))
            {
                using(MemoryStream weightsStream = new MemoryStream())
                {
                    stream.CopyTo(weightsStream);

                    weightsStream.Position = 0;
                    byteArray = weightsStream.ToArray();
                }
            }

            var weights = new double[byteArray.Length / sizeof(double)];
            Buffer.BlockCopy(byteArray, 0, weights, 0, byteArray.Length);
            this.neuralNetwork.LoadWeights(weights);
        }

        public Vector<double> GetLayerNodes(int layer)
        {
            return this.neuralNetwork.nodes[layer].Clone();
        }
    }
}
