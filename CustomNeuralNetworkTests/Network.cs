using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace CustomNeuralNetworkTests
{
	public class Network
	{
		public Network(int[] layerDimensions)
		{
			InitializeParameters(layerDimensions);
		}

		private void InitializeParameters(int[] layerDimensions)
		{
			for (var currentLayer = 1; currentLayer < layerDimensions.Length; currentLayer++)
			{
				LayerWeights.Add(np.random.randn(layerDimensions[currentLayer], layerDimensions[currentLayer - 1]) * .1f);
				LayerBias.Add(np.zeros(new Shape(layerDimensions[currentLayer], 1)));
			}
		}

		public List<NDArray> LayerWeights { get; init; } = new();
		public List<NDArray> LayerBias { get; init; } = new();

		public NDArray ForwardPropagation(NDArray trainingData)
		{
			var activatedValue = trainingData;
			for (int layer = 0; layer < LayerWeights.Count; layer++)
			{
				var previousActivation = activatedValue;
				var linearValue = LinearForward(previousActivation, LayerWeights[layer], LayerBias[layer]);
				LayerForwardValues.Add(linearValue);
				activatedValue = LinearActivationForward(linearValue, layer != LayerWeights.Count - 1 ? Activations.Relu : Activations.Sigmoid);
				LayerForwardActivatedValues.Add(activatedValue);
			}
			return activatedValue;
		}

		public List<NDArray> LayerForwardValues { get; set; } = new();
		public List<NDArray> LayerForwardActivatedValues { get; set; } = new();

		public NDArray LinearForward(NDArray activatedData, NDArray weights, NDArray bias) => np.dot(weights, activatedData) + bias;

		private NDArray LinearActivationForward(NDArray linearForwardValue,
			Activations function)
		{
			NDArray activatedValue;
			if (function == Activations.Sigmoid)
				activatedValue = new Sigmoid().Calculate(linearForwardValue);
			else
				activatedValue = new Relu().Calculate(linearForwardValue);
			return activatedValue;
		}

		public float ComputeCost(NDArray outputActivatedValue, NDArray targets)
		{
			var totalRows = targets.Shape[1];
			var result = (np.multiply(targets, np.log(outputActivatedValue)) + np.multiply(1 - targets, np.log(1 - outputActivatedValue))).flatten();
			return (float ) (-1.0f / totalRows * result.Cast<double>().Sum());
		}
	}
}