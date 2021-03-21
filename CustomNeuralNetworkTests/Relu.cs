using NumSharp;

namespace CustomNeuralNetworkTests
{
	public class Relu : ActivationFunction
	{
		public NDArray Calculate(NDArray value) => np.maximum(0, value);
	}
}