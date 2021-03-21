using NumSharp;

namespace CustomNeuralNetworkTests
{
	public interface ActivationFunction
	{
		public NDArray Calculate(NDArray value);
		public NDArray Derivative(NDArray layerForwardValue);
	}
}