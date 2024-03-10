using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFrame
{
    public sealed class NeuralNetworkConfiguration
    {
        private readonly IContinuousDistribution normalDistribution;

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetworkConfiguration"/> class.
        /// </summary>
        public NeuralNetworkConfiguration()
        {
            this.Layers = new List<Layer>();
            this.normalDistribution = new Normal();
        }

        /// <summary>
        /// Gets or sets a list containing the layers for this configuration.
        /// </summary>
        public List<Layer> Layers { get; }

        /// <summary>
        /// Gets or sets a value indicating whether the result of the neural network should be put though a softmax function.
        /// </summary>
        public bool SoftMaxResult { get; set; }

        internal Vector<double>[] CreateLayeredNodeVector()
        {
            var layerArray = new Vector<double>[this.Layers.Count];
            for (int l = 0; l < this.Layers.Count; l++)
            {
                layerArray[l] = this.Layers[l].CreateVectorForLayer();
            }

            return layerArray;
        }

        internal Matrix<double>[] CreateLayeredWeightMatrix(bool random = false)
        {
            var matrixArray = new Matrix<double>[this.Layers.Count];
            for (int l = 0; l < this.Layers.Count - 1; l++)
            {
                if (random)
                {
                    matrixArray[l] = CreateMatrix.Random<double>((int)this.Layers[l].Nodes, (int)this.Layers[l + 1].Nodes, this.normalDistribution);
                    matrixArray[l] *= Math.Sqrt(1d / this.Layers[l].Nodes); // multiply with a factor for xavier distribution
                }
                else
                {
                    matrixArray[l] = CreateMatrix.Dense<double>((int)this.Layers[l].Nodes, (int)this.Layers[l + 1].Nodes);
                }
            }

            return matrixArray;
        }

        internal int[] CreateLayeredBiasArray()
        {
            var layerArray = new int[this.Layers.Count];
            for (int l = 0; l < this.Layers.Count; l++)
            {
                layerArray[l] = this.Layers[l].HasBias ? 1 : 0;
            }

            return layerArray;
        }

        internal Vector<double>[] CreateLayeredBiasWeightVector(bool random = false)
        {
            var matrixArray = new Vector<double>[this.Layers.Count];
            for (int l = 0; l < this.Layers.Count - 1; l++)
            {
                if (!this.Layers[l].HasBias)
                {
                    CreateVector.Dense<double>(0);
                }
                if (random)
                {
                    matrixArray[l] = CreateVector.Random<double>((int)this.Layers[l + 1].Nodes, this.normalDistribution);
                }
                else
                {
                    matrixArray[l] = CreateVector.Dense<double>((int)this.Layers[l + 1].Nodes);
                }
            }

            return matrixArray;
        }
    }
}
