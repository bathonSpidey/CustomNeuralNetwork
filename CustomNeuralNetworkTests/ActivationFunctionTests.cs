using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;
using NUnit.Framework;

namespace CustomNeuralNetworkTests
{
	public class ActivationFunctionTests
	{
		[Test]
		public void SigmoidShouldHaveValidResults()
		{
			var testMatrix = np.zeros(new Shape(2, 3));
			var function = new Sigmoid();
			var result = function.Calculate(testMatrix);
			Assert.That(result.Shape[0], Is.EqualTo(2));
			Assert.That(result.Shape[1], Is.EqualTo(3));
			Assert.That(result.Data<float>().All(v => Math.Abs(v - .5f) < .01f), Is.True);
		}

		[Test]
		public void ReluShouldHaveValidResults()
		{
			var testMatrix = np.ones(new Shape(2, 3));
			var function = new Relu();
			var result = function.Calculate(testMatrix);
			Assert.That(result.Data<float>().All(v => Math.Abs(v - 1f) < .01f), Is.True);
		}

		[Test]
		public void ReluDerivative()
		{
			var test = new float[,] {{1, -.4f}, {6, -2}};
			var testMatrix = np.array(test);
			var result = new Relu().Derivative(testMatrix);
			Assert.That((float)result[0][0],Is.EqualTo(1f));
			Assert.That((float)result[0][1], Is.EqualTo(0f));
			Assert.That((float)result[1][0], Is.EqualTo(6f));
			Assert.That((float)result[1][1], Is.EqualTo(0f));
		}
	}
}
