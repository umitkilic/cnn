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
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
        int channels=1;
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
        
        for(int i=1;i<4;i++){
            DataSet ds=dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());
        }
    }
}
