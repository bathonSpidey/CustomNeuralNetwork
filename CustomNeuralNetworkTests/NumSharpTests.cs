using System;
using System.Linq;
using NumSharp;
using NUnit.Framework;

namespace CustomNeuralNetworkTests
{
	public class NumSharpTests
	{
		[SetUp]
		public void GetData()
		{
			var array = new[,] {{1, 1}, {0, 0}, {0, 1}, {1, 1}};
			var targetArray = new[,] {{1}, {0}, {0}, {0}};
			data = np.array(array);
			targets = np.array(targetArray);
		}

		private NDArray data;
		private NDArray targets;

		[Test]
		public void DataShapeShouldHaveFourRowsAndTwoColumns()
		{
			Assert.That(data.Shape[0], Is.EqualTo(4));
			Assert.That(data.Shape[1], Is.EqualTo(2));
		}
		[Test]
		public void ReshapingDataShouldInterchangeRowsAndColumns()
		{
			var transposedData=data.transpose();
			var transposeTargets = targets.transpose();
			Assert.That(transposedData.Shape[0], Is.EqualTo(2));
			Assert.That(transposedData.Shape[1], Is.EqualTo(4));
			Assert.That(transposeTargets.Shape[1], Is.EqualTo(4));
		}

		[Test]
		public void InitializingRandomMatricesShouldHaveValidValues()
		{
			var weights = np.random.randn(3, 2)*.1f;
			Assert.That(weights.Shape[0],Is.EqualTo(3));
			Assert.That(weights.Shape[1], Is.EqualTo(2));
			Assert.That(weights.Data<float>().All(v =>  v<=.3f),Is.True);
			Console.WriteLine(weights.ToString());
		}

		[Test]
		public void InitializingZerosShouldBeAllZeros()
		{
			var bias = np.zeros(new Shape(3, 1));
			Assert.That(bias.Shape[0], Is.EqualTo(3));
			Assert.That(bias.Shape[1], Is.EqualTo(1));
			Assert.That(bias.Data<float>().All(v => v == 0f), Is.True);
			Console.WriteLine(bias.ToString());
		}
	}
}