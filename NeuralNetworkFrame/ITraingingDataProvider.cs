namespace NeuralNetworkFrame
{
    public interface ITraingingDataProvider
    {
        double[] GetNextTrainingData();
        double[] GetExpectedResult();
        byte GetExpectedClassification();
    }
}