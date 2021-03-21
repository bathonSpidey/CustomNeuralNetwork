using NumSharp;

namespace CustomNeuralNetworkTests
{
	public class Sigmoid : ActivationFunction
	{
		public NDArray Calculate(NDArray value) => 1/(1 + np.exp(value*-1));
	}
}