using System;
using System.Linq;
using NumSharp;
using NUnit.Framework;

namespace CustomNeuralNetworkTests
{
	public class NetworkTests
	{
		[SetUp]
		public void GetData()
		{
			var array = new[,] { { 1, 1 }, { 0, 0 }, { 0, 1 }, { 1, 0 } };
			var targetArray = new[,] { { 1 }, { 0 }, { 0 }, { 0 } };
			trainingData = np.array(array).transpose();
			targets = np.array(targetArray).transpose();
			network= new Network(new[] { 2, 3, 1 });
		}

		private NDArray trainingData;
		private NDArray targets;
		private Network network;

		[Test]
		public void InitializeWeightsAndBiases()
		{
			Assert.That(network.LayerWeights.Count,Is.EqualTo(2));
			Assert.That(network.LayerWeights[0].Shape[0], Is.EqualTo(3));
			Assert.That(network.LayerWeights[0].Shape[1], Is.EqualTo(2));
			Assert.That(network.LayerWeights[0].Data<float>().All(v => v <= .3f), Is.True);
			Assert.That(network.LayerWeights[1].Shape[0], Is.EqualTo(1));
			Assert.That(network.LayerWeights[1].Shape[1], Is.EqualTo(3));
			Assert.That(network.LayerWeights[1].Data<float>().All(v => v <= .3f), Is.True);
			Assert.That(network.LayerBias[0].Shape[0], Is.EqualTo(3));
			Assert.That(network.LayerBias[1].Shape[0], Is.EqualTo(1));
		}

		[Test]
		public void CalculateLinearForwardForOneLayer()
		{
			var weights = np.random.randn(3, 2) * .1f;
			var bias = np.zeros(new Shape(3, 1));
			var value = network.LinearForward(trainingData,weights,bias);
			Assert.That(value.Shape[0],Is.EqualTo(3));
			Assert.That(value.Shape[1], Is.EqualTo(4));
		}

		[Test]
		public void CalculateOutputLayerActivation()
		{
			var result = GetLastLayerActivation();
			Assert.That(result.Shape[0],Is.EqualTo(1));
			Assert.That(result.Shape[1],Is.EqualTo(4));
			Assert.That(network.LayerForwardActivatedValues.Count,Is.EqualTo(2));
			Assert.That(network.LayerForwardValues.Count, Is.EqualTo(2));
			//return result;
		}

		private NDArray GetLastLayerActivation() => network.ForwardPropagation(trainingData);

		[Test]
		public void ComputeCostShouldHaveValidResults()
		{
			var outputActivations = GetLastLayerActivation();
			var result = network.ComputeCost(outputActivations, targets);
			Assert.That(result,Is.TypeOf<float>());
			Assert.That(result,Is.GreaterThanOrEqualTo(0.069f));
		}

		[Test]
		public void BackPropagationShouldHaveValidGradients()
		{
			network.BackWardPropagation(GetLastLayerActivation(),targets);
		}



	}
}