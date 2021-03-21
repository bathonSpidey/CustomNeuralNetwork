using NumSharp;

namespace CustomNeuralNetworkTests
{
	public class Relu : ActivationFunction
	{
		public NDArray Calculate(NDArray value) => np.maximum(0, value);

		public NDArray Derivative(NDArray layerForwardValue)
		{
			for (var rows = 0; rows < layerForwardValue.Shape[0]; rows++)
			for (var cols = 0; cols < layerForwardValue.Shape[1]; cols++)
					if ((float) layerForwardValue[rows][cols] <= 0f)
						layerForwardValue[rows][cols] = 0f;
			return layerForwardValue;
		}
	}
}