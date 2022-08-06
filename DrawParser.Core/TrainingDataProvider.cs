using NeuralNetworkFrame;
using Serilog;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace DrawParser.Core
{
    public class TrainingDataProvider : ITraingingDataProvider
    {
        private const string TrainingDataFile = "nn/training_data.xml";
        private const string TrainingSetTag = "TrainingSet";
        private const string TrainingDataTag = "TrainingData";
        private const string TrainingDataExpectedTag = "Expected";
        private const string TrainingDataSizeTag = "Size";

        private List<TrainingData> trainingData;
        private readonly Stopwatch epochWatch;
        private readonly ILogger logger;
        private int currentTrainingDataIndex;
        private int epochCounter;

        public TrainingDataProvider()
        {
            this.currentTrainingDataIndex = 0;
            this.epochCounter = 1;
            this.trainingData = new List<TrainingData>();
            this.epochWatch = new Stopwatch();
            this.logger = new LoggerConfiguration()
                .WriteTo.Debug()
                .WriteTo.File("nn/TrainingDataProviderlog.txt", outputTemplate: "{Message:lj}{NewLine}", rollOnFileSizeLimit: true, fileSizeLimitBytes: 996147200)
                .CreateLogger();
        }

        public void AddTrainingData(TrainingData trainingData)
        {
            _ = trainingData ?? throw new ArgumentNullException(nameof(trainingData));

            this.trainingData.Add(trainingData);
        }

        public void SaveTrainingSet()
        {
            this.trainingData.Sort();
            var trainingSet = new XElement(TrainingSetTag);

            StringBuilder sb = new StringBuilder();
            foreach (var td in this.trainingData)
            {
                sb.Clear();
                var byteArray = new byte[td.Data.Length * sizeof(double)];
                Buffer.BlockCopy(td.Data, 0, byteArray, 0, byteArray.Length);
                for(int i = 0; i < byteArray.Length; i++)
                {
                    if (byteArray[i] > 0)
                    {
                        sb.Append($"{i}:{byteArray[i].ToString("X2")};");
                    }
                }

                var data = new XElement(TrainingDataTag, sb.ToString());
                data.Add(new XAttribute(TrainingDataExpectedTag, td.Expected));
                data.Add(new XAttribute(TrainingDataSizeTag, byteArray.Length));
                trainingSet.Add(data);
            }

            if (!Directory.Exists(Path.GetDirectoryName(TrainingDataFile)))
            {
                if (Path.GetDirectoryName(TrainingDataFile) is String dir)
                Directory.CreateDirectory(dir);
            }

            trainingSet.Save(TrainingDataFile);
        }

        public void LoadTrainingSet()
        {
            if (!File.Exists(TrainingDataFile))
            {
                return;
            }

            var trainingSets = XDocument.Load(TrainingDataFile);

            if (trainingSets == null)
            {
                return;
            }

            var trainingSet = trainingSets?.Root?.Elements(TrainingDataTag);
            if (trainingSet == null)
            {
                return;
            }

            foreach (var trainingData in trainingSet)
            {
                var data = this.ParseTrainingData(trainingData);
                if (data != null)
                {
                    this.trainingData.Add(data);
                }
            }
        }

        public int GetTrainingDataAmount(int expectedInTraingData)
        {
            return this.trainingData.Where(t => t.Expected.Equals(expectedInTraingData)).Count();
        }

        private TrainingData? ParseTrainingData(XElement element)
        {
            int? expectedData = null;
            int? size = null;
            if (element.Attribute(TrainingDataExpectedTag) is XAttribute TrainingDataExpectedAttribute)
            {
                expectedData = Convert.ToInt32(TrainingDataExpectedAttribute.Value);
            }

            if (element.Attribute(TrainingDataSizeTag) is XAttribute TrainingDataSizeAttribute)
            {
                size = Convert.ToInt32(TrainingDataSizeAttribute.Value);
            }

            if (!expectedData.HasValue || !size.HasValue)
            {
                return null;
            }

            var byteArray = new byte[size.Value];
            string dataString = element.Value.ToString();
            string[] data = dataString.Split(';');
            for (int i = 0; i < data.Length; i++)
            {
                if (!String.IsNullOrEmpty(data[i]))
                {
                    var dataParts = data[i].Split(':');
                    var index = Convert.ToInt32(dataParts[0]);
                    var dataValue = Convert.ToByte(dataParts[1], 16);
                    byteArray[index] = dataValue;
                }
            }

            var doubleArray = new double[byteArray.Length / sizeof(double)];
            Buffer.BlockCopy(byteArray, 0, doubleArray, 0, byteArray.Length);

            return new TrainingData(doubleArray, expectedData.Value);
        }

        public double[] GetNextTrainingData()
        {
            if (this.currentTrainingDataIndex == 0)
            {
                if (this.epochWatch.IsRunning)
                {
                    this.epochWatch.Stop();
                    this.logger.Information($"Epoch nr{this.epochCounter} finished: {this.epochWatch.ElapsedMilliseconds} ms");
                    this.epochCounter++;
                    this.epochWatch.Reset();
                }

                this.epochWatch.Start();
            }

            var data = this.trainingData[this.currentTrainingDataIndex].Data;
            this.currentTrainingDataIndex++;
            if (currentTrainingDataIndex >= this.trainingData.Count)
            {
                // end of epoch
                this.currentTrainingDataIndex = 0;
            }

            return data;
        }

        public double[] GetExpectedResult()
        {
            var expectedResult = new double[10];
            expectedResult[this.trainingData[this.currentTrainingDataIndex].Expected] = 1f;
            return expectedResult;
        }

        public byte GetExpectedClassification()
        {
            return (byte)this.trainingData[this.currentTrainingDataIndex].Expected;
        }
    }
}
