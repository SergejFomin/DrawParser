using MathNet.Numerics.LinearAlgebra;
using Serilog;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFrame
{
    public sealed class NeuralNetworkTrainer
    {
        private const double LearningRateAcceleration = 1.1d;
        private const double InitialLearningRate = 0.001d;

        private readonly ILogger logger;
        private NeuralNetwork neuralNetwork;
        private ITraingingDataProvider traingingDataProvider;
        private double totalError;
        private int errorCount;

        internal NeuralNetworkTrainer(NeuralNetwork neuralNetwork, ITraingingDataProvider traingingDataProvider)
        {
            MathNet.Numerics.Control.UseMultiThreading();
            this.neuralNetwork = neuralNetwork ?? throw new ArgumentNullException(nameof(neuralNetwork));
            this.traingingDataProvider = traingingDataProvider ?? throw new ArgumentNullException(nameof(traingingDataProvider));
            this.totalError = 0.0f;
            this.errorCount = 0;

            this.logger = new LoggerConfiguration()
                .WriteTo.Debug()
                .WriteTo.File("nn/log.txt", outputTemplate: "{mm:ss.fffffff} {Message:lj}{NewLine}", rollOnFileSizeLimit: true, fileSizeLimitBytes:996147200)
                .CreateLogger();
        }

        public void StartTraining()
        {
            // get training data set
            var trainingData = this.traingingDataProvider.GetNextTrainingData();
            var expectedResult = CreateVector.Dense<double>(this.traingingDataProvider.GetExpectedResult());
            var expectedClassification = this.traingingDataProvider.GetExpectedClassification();
            var result = CreateVector.Dense<double>(this.neuralNetwork.CalculateOutput(trainingData));
            result = this.neuralNetwork.GetLayerNodes(this.neuralNetwork.configuration.Layers.Count - 1);
            var error = FunctionProvider.MeanSquaredError(result, expectedResult);

            this.AddError(error);
            var message = String.Format($"{expectedClassification},{error.ToString(CultureInfo.InvariantCulture)},{this.GetAverageError().ToString(CultureInfo.InvariantCulture)}");
            this.logger.Information(message);

            // prepare arrays for new derivative values and new weigh values
            var weightDerivatives = this.neuralNetwork.configuration.CreateLayeredWeightMatrix();
            var biasWeightDerivatives = this.neuralNetwork.configuration.CreateLayeredBiasWeightVector();
            var nodesBaseDerivatives = this.neuralNetwork.configuration.CreateLayeredNodeVector();

            // get current weights and node values
            var currentWeights = this.neuralNetwork.neuralNetwork.weights;
            var currentBiasWeights = this.neuralNetwork.neuralNetwork.biasWeights;
            var currentWeightedSums = this.neuralNetwork.neuralNetwork.weightedSums;
            var currentNodeValues = this.neuralNetwork.neuralNetwork.nodes;
            var currentBiasValues = this.neuralNetwork.neuralNetwork.bias;

            // back propagation (calculate for each weights it's impact on the final error)
            var layers = this.neuralNetwork.configuration.Layers;
            var lastLayerIndex = layers.Count - 1;
            nodesBaseDerivatives[layers.Count - 1] = result - expectedResult;

            weightDerivatives[lastLayerIndex - 1] = nodesBaseDerivatives[lastLayerIndex].OuterProduct(currentNodeValues[lastLayerIndex - 1]).Transpose();
            currentWeights[lastLayerIndex - 1] -= InitialLearningRate * weightDerivatives[lastLayerIndex - 1];
            biasWeightDerivatives[lastLayerIndex - 1] -= InitialLearningRate * nodesBaseDerivatives[lastLayerIndex];

            for (int l = layers.Count - 2; l > 0; l--)
            {
                var derivedActivationFunction = FunctionProvider.GetFunctionDerivative(layers[l].ActivationFunction);
                nodesBaseDerivatives[l] = currentWeightedSums[l].Map(derivedActivationFunction);
                nodesBaseDerivatives[l] = nodesBaseDerivatives[l].PointwiseMultiply(currentWeights[l] * nodesBaseDerivatives[l + 1]);

                weightDerivatives[l - 1] = nodesBaseDerivatives[l].OuterProduct(currentNodeValues[l - 1]).Transpose();
                currentWeights[l - 1] -= InitialLearningRate * weightDerivatives[l - 1];
                biasWeightDerivatives[l - 1] -= InitialLearningRate * nodesBaseDerivatives[l];
            }
        }

        private void AddError(double error)
        {
            this.errorCount++;
            this.totalError += error;

            if (this.errorCount >= 100)
            {
                this.totalError = this.GetAverageError();
                this.errorCount = 1;
            }
        }

        public double GetAverageError()
        {
            return this.totalError/this.errorCount;
        }

        private double[][][] CreateWeightDerivativesArray()
        {
            // back propagation (calculate for each weights what it's impact is on the final error)
            var derivatives = new double[this.neuralNetwork.configuration.Layers.Count - 1][][];

            // first dimension is as big as there are layers of connections (layers - 1) (amount of input layers)
            for (int l = 0; l < derivatives.Length; l++)
            {
                // every second dimension is as big as the amount of nodes in each input layers (0 to length - 1)
                derivatives[l] = new double[this.neuralNetwork.configuration.Layers[l].Nodes][];

                // every third dimension is as big as the amount of nods in each output layers (1 to length)
                for (int i = 0; i < this.neuralNetwork.configuration.Layers[l].Nodes; i++)
                {
                    derivatives[l][i] = new double[this.neuralNetwork.configuration.Layers[l + 1].Nodes];
                }
            }

            return derivatives;
        }

        private double[][] CreateOutputNodesArray()
        {
            // fill layers array and data
            var layersCount = this.neuralNetwork.configuration.Layers.Count;
            var layers = new uint[layersCount];
            for (int l = 0; l < layersCount; l++)
            {
                layers[l] = this.neuralNetwork.configuration.Layers[l].Nodes;
            }

            // first dimension is as big as there are layers
            var nodes = new double[layersCount][];

            // second dimensions are as big as there are nodes in each layer
            // first layer does not need any nodes (is not an output node)
            for (int l = 1; l < layersCount; l++)
            {
                nodes[l] = new double[layers[l]];
            }

            return nodes;
        }

        private double[][] CreateBiasWeightDerivativesArray()
        {
            // set size for bias weights array
            var biasWeightsDerivatives = this.CreateBiasWeightsArray();
            biasWeightsDerivatives[this.neuralNetwork.configuration.Layers.Count - 2] = new double[1];

            return biasWeightsDerivatives;
        }

        private double[][] CreateBiasWeightsArray()
        {
            // fill layers array and data
            var layersCount = this.neuralNetwork.configuration.Layers.Count;
            var layers = new uint[layersCount];
            for (int l = 0; l < layersCount; l++)
            {
                layers[l] = this.neuralNetwork.configuration.Layers[l].Nodes;
            }

            // set size for bias weights array
            var biasWeights = new double[layersCount - 1][];
            for (int l = 0; l < biasWeights.Length; l++)
            {
                biasWeights[l] = new double[layers[l + 1]];
            }

            return biasWeights;
        }
    }
}
