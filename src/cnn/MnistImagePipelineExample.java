/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cnn;

import java.io.File;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author ferhat
 */
public class MnistImagePipelineExample {
    private static Logger log=LoggerFactory.getLogger(MnistImagePipelineExample.class);
    
    public static void main(String[] args) throws Exception{
        int height=28;
        int width=28;
        int channels=3;
        int rngseed=123;
        Random randNumGen=new Random(rngseed);
        int batchSize=1;
        int outputNum=2; // possible output number
        
        File trainData=new File("data/training_data");
        File testData=new File("data/testing_data");
        
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        ParentPathLabelGenerator labelmaker=new ParentPathLabelGenerator();
        
        ImageRecordReader recorder=new ImageRecordReader(height,width,channels,labelmaker);
        
        recorder.initialize(train);
        recorder.setListeners(new LogRecordListener());
        
        DataSetIterator dataIter=new RecordReaderDataSetIterator(recorder,batchSize,1,outputNum);
        
        recorder.initialize(test);
        recorder.setListeners(new LogRecordListener());
        
        DataSetIterator dataIterTest=new RecordReaderDataSetIterator(recorder,batchSize,1,outputNum);
        
        /*for(int i=1;i<4;i++){
            DataSet ds=dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());
        }*/
        
        // input elimizde
        
        int iterations=1;
        
        MultiLayerConfiguration conf=new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5,5)
                        .nIn(channels)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5,5)
                          .stride(1,1)
                          .nOut(50)
                           .activation(Activation.IDENTITY)
                            .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                            .nOut(500)
                            .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 3))
                .backprop(true).pretrain(false).build();
        
        
        
        // layer conf
        
        MultiLayerNetwork model=new MultiLayerNetwork(conf);
        model.init();
        
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < 1; i++) {
            model.fit(dataIter);
            
            Evaluation eval=new Evaluation(outputNum);
            while (dataIterTest.hasNext()) {
                    DataSet ds=dataIterTest.next();
                    INDArray output=model.output(ds.getFeatureMatrix(),false);
                    eval.eval(ds.getLabels(), output);
            }
            
            //log.info(eval.stats());
            System.out.println(eval.stats());
            dataIterTest.reset();
        }
        
    }
}