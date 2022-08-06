using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using DrawParser.Core;
using NeuralNetworkFrame;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace WpfAppdrawtest
{
    public sealed class MainWindowViewModel : ObservableObject
    {
        private const int TitleUpdateTimer = 1000;
        private const string WeightsAndBiases = "nn/weights.wab";
        private const string ApplicationName = "DrawParser";
        private readonly NeuralNetwork neuralNetwork;
        private readonly NeuralNetworkTrainer neuralNetworkTrainer;
        private readonly TrainingDataProvider trainingDataProvider;
        private readonly Timer titleUpdateTimer;

        private CancellationTokenSource? trainingCancellation;
        private int expectedTrainingData;
        private double[] imageBrightnessData;
        private string titleAddition;

        private bool isTraining;

        public MainWindowViewModel()
        {
            this.isTraining = false;
            this.titleAddition = String.Empty;

            this.PropertyChanged += this.OwnPropertyChanged;
            this.trainingDataProvider = new TrainingDataProvider();
            this.trainingDataProvider.LoadTrainingSet();

            this.imageBrightnessData = Array.Empty<double>();
            this.ClassificationsCollection = new ObservableCollection<ClassificationViewModel>();

            var nnConfig = new NeuralNetworkConfiguration();
            nnConfig.Layers.Add(new Layer(28 * 28, true, Functions.None));
            nnConfig.Layers.Add(new Layer(14, true, Functions.Sigmoid));
            nnConfig.Layers.Add(new Layer(10, true, Functions.Sigmoid));
            nnConfig.SoftMaxResult = false;

            this.neuralNetwork = new NeuralNetwork(nnConfig);
            this.neuralNetwork.LoadWeightsAndBias(WeightsAndBiases);

            this.neuralNetworkTrainer = this.neuralNetwork.GetNeuralNetworkTrainer(this.trainingDataProvider);

            this.AddTrainingDataCommand = new RelayCommand(this.AddTrainingData);
            this.StartTrainingCommand = new RelayCommand(this.StartTraining);
            this.StopTrainingCommand = new RelayCommand(this.StopTraining);

            this.titleUpdateTimer = new Timer((param) =>
            {
                this.UpdateApplicationTitle();
            }, null, 0, Timeout.Infinite);
        }

        public String Title
        {
            get
            {
                return $"{ApplicationName} - {this.titleAddition}";
            }
        }

        public RelayCommand AddTrainingDataCommand { get; }

        public RelayCommand StartTrainingCommand { get; }

        public RelayCommand StopTrainingCommand { get; }

        public Action? ClearCanvasAction { get; set; }

        public double[] ImageBrightnessData
        {
            get
            {
                return imageBrightnessData;
            }

            set
            {
                imageBrightnessData = value;
                this.OnPropertyChanged();
            }
        }

        public int ExpectedTrainingData
        {
            get
            {
                return expectedTrainingData;
            }

            set
            {
                expectedTrainingData = value;
                this.OnPropertyChanged();
                this.OnPropertyChanged(nameof(this.AmountCurrentTrainingData));
            }
        }

        public int AmountCurrentTrainingData
        {
            get
            {
                return this.trainingDataProvider.GetTrainingDataAmount(this.ExpectedTrainingData);
            }
        }

        public ObservableCollection<ClassificationViewModel> ClassificationsCollection { get; private set; }

        public void OnClosing()
        {
            this.trainingDataProvider.SaveTrainingSet();
            this.neuralNetwork.SaveWeightsAndBias(WeightsAndBiases);
        }

        private void UpdateClassification()
        {
            var wasTraining = this.isTraining;
            if (wasTraining)
            {
                this.StopTraining();
            }

            var result = this.neuralNetwork.CalculateOutput(this.ImageBrightnessData);
            var classifications = new ClassificationViewModel[result.Length];

            for (int i = 0; i < result.Length; i++)
            {
                classifications[i] = new ClassificationViewModel(i, result[i] * 100f);
            }

            this.ClassificationsCollection = new ObservableCollection<ClassificationViewModel>(classifications.OrderByDescending(c => c.Probability));
            this.OnPropertyChanged(nameof(this.ClassificationsCollection));

            if (wasTraining)
            {
                this.StartTraining();
            }
        }

        private void AddTrainingData()
        {
            this.trainingDataProvider.AddTrainingData(new TrainingData(this.ImageBrightnessData, this.ExpectedTrainingData));
            this.ClearCanvasAction?.Invoke();
            this.OnPropertyChanged(nameof(this.AmountCurrentTrainingData));
        }

        private void StartTraining()
        {
            if (this.isTraining)
            {
                this.StopTraining();
            }

            this.trainingCancellation = new CancellationTokenSource();
            Task.Run(() =>
            {
                while (!this.trainingCancellation.Token.IsCancellationRequested)
                {
                    this.neuralNetworkTrainer.StartTraining();
                }
            },this.trainingCancellation.Token);

            this.isTraining = true;
            this.titleUpdateTimer.Change(3, TitleUpdateTimer);
            this.UpdateApplicationTitle();
        }

        private void StopTraining()
        {
            this.trainingCancellation?.Cancel();
            this.isTraining = false;
            this.titleUpdateTimer.Change(Timeout.Infinite, Timeout.Infinite);
            this.UpdateApplicationTitle();
        }

        private void UpdateApplicationTitle()
        {
            if (this.isTraining)
            {
                this.titleAddition = $"Training: Average Error: {this.neuralNetworkTrainer.GetAverageError()}";
            }
            else
            {
                this.titleAddition = String.Empty;
            }

            this.OnPropertyChanged(nameof(this.Title));
        }

        private void OwnPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e?.PropertyName == nameof(this.ImageBrightnessData))
            {
                this.UpdateClassification();
            }
        }
    }
}
