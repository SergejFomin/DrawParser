using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFrame
{
    public sealed class Layer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Layer"/> class.
        /// </summary>
        /// <param name="nodes">The nodes.</param>
        /// <param name="hasBias">if set to <c>true</c> this layer will have a bias.</param>
        /// <param name="activationFunction">The activation function.</param>
        public Layer(uint nodes, bool hasBias, Functions activationFunction)
        {
            this.Nodes = nodes;
            this.HasBias = hasBias;
            this.ActivationFunction = activationFunction;
        }

        /// <summary>
        /// Gets the amount of nodes this layer consists.
        /// </summary>
        public uint Nodes { get; }

        /// <summary>
        /// Gets a value indicating whether this layer has a bias.
        /// </summary>
        public bool HasBias { get; }

        /// <summary>
        /// Gets the activation function.
        /// </summary>
        public Functions ActivationFunction { get; }

        internal Vector<double> CreateVectorForLayer()
        {
            return CreateVector.Dense<double>((int)this.Nodes);
        }
    }
}
