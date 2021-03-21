using NumSharp;

namespace CustomNeuralNetworkTests
{
	public class Sigmoid : ActivationFunction
	{
		public NDArray Calculate(NDArray value) => 1/(1 + np.exp(value*-1));

		public NDArray Derivative(NDArray layerForwardValue) => Calculate(layerForwardValue) * (1 - Calculate(layerForwardValue));
	}
}