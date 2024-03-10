using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkFrame
{
    internal class InternalNeuralNetwork
    {
        internal int layersCount;

        // hold a count of nodes for each layer
        internal uint[] layers;

        // 1st dimension is for each layer of nodes
        // 2nd dimension is the single nodes of the layer
        internal Vector<double>[] nodes;

        // 1st dimension is for each layer of nodes
        // 2nd dimension is the single nodes of the layer
        // weighted sums hold the values before they get passed through the activation function
        internal Vector<double>[] weightedSums;

        // 1st dimension is for each layer of weights between node layers.
        // 2nd dimension is for each node on the left(input) side.
        // 3rd dimension is for each node on the right(output) side.
        // each input node has a connection to each output node in the given weights layer
        internal Matrix<double>[] weights;

        // the bias for each output layer
        internal int[] bias;

        // 1st dimension is for each layer of weights between node layers.
        // 2rd dimension is for each node on the right(output) side.
        internal Vector<double>[] biasWeights;

        // 1st dimension is for each layer. Each layer can have a different activation function
        internal Func<double, double>[] activateFunction;

        internal InternalNeuralNetwork(NeuralNetworkConfiguration configuration, bool randomWeights)
        {
            MathNet.Numerics.Control.UseMultiThreading();
            _ = configuration ?? throw new ArgumentNullException(nameof(configuration));

            // fill layers array and data and activation functions
            this.layersCount = configuration.Layers.Count;
            this.layers = new uint[this.layersCount];
            this.activateFunction = new Func<double, double>[this.layersCount];

            for (int l = 0; l < this.layersCount; l++)
            {
                this.layers[l] = configuration.Layers[l].Nodes;
                this.activateFunction[l] = FunctionProvider.GetFunction(configuration.Layers[l].ActivationFunction);
            }

            // set bias values for each output layer
            this.bias = configuration.CreateLayeredBiasArray();

            // set size for bias weights array
            this.biasWeights = configuration.CreateLayeredBiasWeightVector(randomWeights);

            // first dimension is as big as there are layers
            this.nodes = configuration.CreateLayeredNodeVector();
            this.weightedSums = configuration.CreateLayeredNodeVector();
            this.weightedSums[0] = CreateVector.Dense<double>(0); // 0 element because the first vector is not needed

            // first dimension is as big as there are layers of connections (layers - 1) (amount of input layers)
            this.weights = configuration.CreateLayeredWeightMatrix(randomWeights);
        }

        internal Vector<double> CalculateOutput(double[] inputValues)
        {
            return this.CalculateOutput(CreateVector.DenseOfArray<double>(inputValues));
        }

        internal Vector<double> CalculateOutput(Vector<double> inputValues)
        {
            if (inputValues.Count != this.nodes[0].Count)
            {
                throw new ArgumentException($"Array should consist of {this.nodes[0].Count} values.", nameof(inputValues));
            }

            // copy Vector
            inputValues.CopyTo(this.nodes[0]);

            // iterate though each output/hidden layer
            for (int l = 0; l < this.layersCount - 1; l++)
            {
                // Multiply the weights matrix with the input vector to gain the sum of the weighted inputs in the weights sums vector
                this.weights[l].Transpose().Multiply(this.nodes[l], this.weightedSums[l + 1]);

                if (this.bias[l] > 0)
                {
                    this.weightedSums[l + 1] += this.biasWeights[l];
                }

                for (int i = 0; i < this.weightedSums[l + 1].Count; i++)
                {
                    this.nodes[l + 1][i] = this.activateFunction[l + 1](this.weightedSums[l + 1][i]);
                }
            }

            return this.nodes[this.layersCount - 1];
        }

        internal double[] GetFlatWeights()
        {
            uint weightCount = 0;
            for (int l = 1; l < this.layersCount; l++)
            {
                weightCount += this.layers[l- 1] * this.layers[l];
                weightCount += this.layers[l]; // add output nodes again for bias weights
            }

            var flatWeights = new double[weightCount];
            uint flatWeightsIndex = 0;
            for (int l = 0; l < this.layersCount - 1; l++)
            {
                for (int o = 0; o < this.layers[l + 1]; o++)
                {
                    for (int i = 0; i < this.layers[l]; i++)
                    {
                        flatWeights[flatWeightsIndex] = this.weights[l][i,o];
                        flatWeightsIndex++;
                    }

                    flatWeights[flatWeightsIndex] = this.biasWeights[l][o];
                    flatWeightsIndex++;
                }
            }

            return flatWeights;
        }

        internal void LoadWeights(double[] flatWeights)
        {
            uint flatWeightsIndex = 0;
            for (int l = 0; l < this.layersCount - 1; l++)
            {
                for (int o = 0; o < this.layers[l + 1]; o++)
                {
                    for (int i = 0; i < this.layers[l]; i++)
                    {
                        this.weights[l][i,o] = flatWeights[flatWeightsIndex];
                        flatWeightsIndex++;
                    }

                    this.biasWeights[l][o] = flatWeights[flatWeightsIndex];
                    flatWeightsIndex++;
                }
            }
        }
    }
}