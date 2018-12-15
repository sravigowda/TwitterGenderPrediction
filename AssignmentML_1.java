import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.LogManager;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.RegExpReplace;
import org.apache.spark.sql.catalyst.expressions.RegExpExtract;
import org.apache.spark.sql.types.DataTypes;

import com.sun.org.apache.regexp.internal.RE;
import com.vdurmont.emoji.EmojiParser;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.regexp_replace;
import static org.apache.spark.sql.functions.regexp_extract;
import static org.apache.spark.sql.functions.lower;
import static org.apache.spark.sql.functions.concat;
import org.apache.spark.sql.functions;


import scala.collection.mutable.WrappedArray;

import static org.apache.spark.sql.functions.col;


public class AssignmentML {

public static void main (String[] args) {
		
		
Logger.getLogger("org").setLevel(Level.ERROR);
			
		SparkSession sparkSession = SparkSession.builder().appName("AssignmentML").master("local[*]").getOrCreate();
		
				
		Dataset<Row> TwitterData  = sparkSession.read().option("header", true).option("inferschema", true).option("mode","DROPMALFORMED").csv(args[0]);
		
		Dataset<Row> NewTwitterData = TwitterData.select(col("_unit_id").cast("integer"),col("gender"),
										col("gender:confidence").cast("float"),
										col("text"),col("retweet_count").cast("Integer"),
										col("description"), col("fav_number").cast("Integer")); 
		
		//NewTwitterData.printSchema();
		//System.out.println(NewTwitterData.count());
		
		NewTwitterData.describe().show();
				
		NewTwitterData = NewTwitterData.filter(col("gender:confidence").equalTo("1.0"))
				.filter((col("gender").contains("male")).or(col("gender").contains("brand")).or(col("gender").contains("female")))
				.filter(col("text").isNotNull())
				.filter(col("description").isNotNull());
		
		//System.out.println(NewTwitterData.count());
		//NewTwitterData.show();
		

		
		StringIndexerModel labelindexer = new StringIndexer()
				.setInputCol("gender")
				.setOutputCol("label").fit(NewTwitterData);
		
		
		Dataset<Row> Gender_label = labelindexer.transform(NewTwitterData);
		Gender_label.show();
		NewTwitterData = labelindexer.transform(NewTwitterData);
		
		NewTwitterData.show();
		
		
		NewTwitterData = NewTwitterData.withColumn("text", regexp_replace(NewTwitterData.col("text"),"<[^>]*>",""));
		NewTwitterData = NewTwitterData.withColumn("emoticons", regexp_extract(NewTwitterData.col("text"),"(\\:\\w+\\:|\\<[\\/\\\\]?3|[\\(\\)\\\\\\D|\\*\\$][\\-\\^]?[\\:\\;\\=]|[\\:\\;\\=B8][\\-\\^]?[3DOPp\\@\\$\\*\\\\\\)\\(\\/\\|])(?=\\s|[\\!\\.\\?]|$)",1));
		
		NewTwitterData = NewTwitterData.withColumn("text", regexp_replace(NewTwitterData.col("text"),"[\\W]"," "));
		NewTwitterData = NewTwitterData.withColumn("text", lower(col("text")));
		
		
		NewTwitterData = NewTwitterData.withColumn("description", regexp_replace(NewTwitterData.col("description"),"[^ 'a-zA-Z0-9,.?!]",""));
		
		NewTwitterData = NewTwitterData.withColumn("description", regexp_replace(NewTwitterData.col("description"),"[\\W]"," "));
		NewTwitterData = NewTwitterData.withColumn("description", lower(col("description")));
		
		
		
		Tokenizer tokenizer_text = new Tokenizer().setInputCol("text").setOutputCol("text_words");
		NewTwitterData = tokenizer_text.transform(NewTwitterData);
		
		
		Tokenizer tokenizer_desc  = new Tokenizer().setInputCol("description").setOutputCol("description_words");
		NewTwitterData = tokenizer_desc.transform(NewTwitterData);
		
		
		StopWordsRemover remover_text = new StopWordsRemover().setInputCol("text_words").setOutputCol("text_remove");
		NewTwitterData = remover_text.transform(NewTwitterData);
		
		
		StopWordsRemover remover_desc = new StopWordsRemover().setInputCol("description_words").setOutputCol("description_remove");
		NewTwitterData = remover_desc.transform(NewTwitterData);
		
		
		Stemmer stem_text = new Stemmer().setInputCol("text_remove").setOutputCol("text_stem");
		NewTwitterData = stem_text.transform(NewTwitterData);
		
		Stemmer stem_desc = new Stemmer().setInputCol("description_remove").setOutputCol("description_stem");
		NewTwitterData = stem_desc.transform(NewTwitterData);

		Tokenizer tokenizer_emot = new Tokenizer().setInputCol("emoticons").setOutputCol("emoticons_split");
		NewTwitterData = tokenizer_emot.transform(NewTwitterData);
		
		
		HashingTF hashingTF_text = new HashingTF()
				.setNumFeatures(1000)
				.setInputCol("text_stem")
				.setOutputCol("numFeatures_text");
		NewTwitterData = hashingTF_text.transform(NewTwitterData);
		
		HashingTF hashingTF_desc = new HashingTF()
				.setNumFeatures(1000)
				.setInputCol("description_stem")
				.setOutputCol("numFeatures_desc");
		NewTwitterData = hashingTF_desc.transform(NewTwitterData);
		
		HashingTF hashingTF_emot = new HashingTF()
				.setNumFeatures(1000)
				.setInputCol("emoticons_split")
				.setOutputCol("numFeatues_emoticons");
		NewTwitterData = hashingTF_emot.transform(NewTwitterData);
		
		
		IDF idf_text = new IDF()
					   .setInputCol("numFeatures_text")
					   .setOutputCol("IDF_text");
		IDFModel idfmodel_text = idf_text.fit(NewTwitterData);
		NewTwitterData = idfmodel_text.transform(NewTwitterData);
		
		IDF idf_desc = new IDF()
				   .setInputCol("numFeatures_desc")
				   .setOutputCol("IDF_desc");
	    IDFModel idfmodel_desc = idf_desc.fit(NewTwitterData);
	    NewTwitterData = idfmodel_desc.transform(NewTwitterData);
	
	
	    IDF idf_emot = new IDF()
				   .setInputCol("numFeatues_emoticons")
				   .setOutputCol("IDF_emot");
	    IDFModel idfmodel_emot = idf_emot.fit(NewTwitterData);
	    NewTwitterData = idfmodel_emot.transform(NewTwitterData);

		
	VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"IDF_text","IDF_desc","retweet_count", "IDF_emot",
            		"fav_number"})
            .setOutputCol("features");
	
	Dataset<Row> LRdf = assembler.transform(NewTwitterData).select("label","features");    
    LRdf.show();
	
    Dataset<Row>[] splits = LRdf.randomSplit(new double[]{0.7, 0.3},46L);
    Dataset<Row> trainingData = splits[0];		//Training Data
    Dataset<Row> testData = splits[1];			//Testing Data
    
	DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
			.setSeed(0).setMaxDepth(12).setMaxBins(6);
	
	RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setSeed(0)
			.setMaxDepth(30).setNumTrees(30);
	

	
	
	DecisionTreeClassificationModel Model = dt.fit(trainingData);	

	RandomForestClassificationModel RFModel = rf.fit(trainingData);
	
	
	
	

	//System.out.println("Learned Decision tree" + Model.toDebugString());
	
	IndexToString labelConverter = new IndexToString().setInputCol("label").setOutputCol("labelStr")
			.setLabels(labelindexer.labels());
	
	IndexToString predConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictionStr")
			.setLabels(labelindexer.labels());

	
	/**************** Decision Tree Model transforming Test Data **************************/
	Dataset<Row> rawPredictions = Model.transform(testData);
	Dataset<Row> predictions = predConverter.transform(labelConverter.transform(rawPredictions));
	//predictions.select("predictionStr", "labelStr", "features").show();
		
	/*************************Decision Tree Model Evaluation for Testing Data *********************/
	// View confusion matrix
	System.out.println("Confusion Matrix for Decision Tree (Testing Data):");
	predictions.groupBy(col("labelStr"), col("predictionStr")).count().show();

	// Accuracy computation
	MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
			.setPredictionCol("prediction").setMetricName("accuracy"); 
	double accuracy = evaluator.evaluate(predictions);
	System.out.println("Decision Tree Accuracy  for Testing data = " + Math.round(accuracy * 100) + " %");
	
	double precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions);
	System.out.println("Decision Tree Precision  for Testing data = " + Math.round(precision*100) + " %");
	
	double recall = evaluator.setMetricName("weightedRecall").evaluate(predictions);
	System.out.println("Decision Tree Recall for Testing data  = " + Math.round(recall*100) + " %");
	
	
	/**************** Decision Tree Model transforming Training Data **************************/
	
	Dataset<Row> trainingrawPredictions = Model.transform(trainingData);
	Dataset<Row> trainingpredictions = predConverter.transform(labelConverter.transform(trainingrawPredictions));
	//predictions.select("predictionStr", "labelStr", "features").show();
		
	/*************************Decision Tree Model Evaluation for Training Data *********************/
	// View confusion matrix
	System.out.println("Confusion Matrix for Decision Tree (Training Data):");
	trainingpredictions.groupBy(col("labelStr"), col("predictionStr")).count().show();

	// Accuracy computation
	MulticlassClassificationEvaluator trainingevaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
			.setPredictionCol("prediction").setMetricName("accuracy"); 
	double trainingaccuracy = trainingevaluator.evaluate(trainingpredictions);
	System.out.println("Decision Tree Accuracy for training Data = " + Math.round(trainingaccuracy * 100) + " %");
	
	double trainingprecision = evaluator.setMetricName("weightedPrecision").evaluate(trainingpredictions);
	System.out.println("Decision Tree Precision = " + Math.round(trainingprecision*100) + " %");
	
	double trainingrecall = evaluator.setMetricName("weightedRecall").evaluate(trainingpredictions);
	System.out.println("Decision Tree Recall = " + Math.round(trainingrecall*100) + " %");
	
	/**************************Random Forest Tree Model transforming Test Data**************************************/
	
	Dataset<Row> RFrawPredictions = RFModel.transform(testData);
	Dataset<Row> RFpredictions = predConverter.transform(labelConverter.transform(RFrawPredictions));
	//RFpredictions.select("predictionStr", "labelStr", "features").show();
	
	/*************************Random Forest Tree Model Evaluation for Testing Data *********************/
	// View confusion matrix
	System.out.println("Confusion Matrix for Random Forest Tree:");
	RFpredictions.groupBy(col("labelStr"), col("predictionStr")).count().show();

	// Accuracy computation
	
	
	double RFaccuracy = evaluator.evaluate(RFpredictions);
	System.out.println("Random Forest Accuracy of Testing Data = " + Math.round(RFaccuracy * 100) + " %");
	
	double RFprecision = evaluator.setMetricName("weightedPrecision").evaluate(RFpredictions);
	System.out.println("Decision Tree Precision of Testing Data = " + Math.round(RFprecision*100) + " %");
	
	double RFrecall = evaluator.setMetricName("weightedRecall").evaluate(RFpredictions);
	System.out.println("Decision Tree Recall of Testing Data = " + Math.round(RFrecall*100) + " %");

	
/**************************Random Forest Model transforming Training Data**************************************/
	
	Dataset<Row> TrainingRFrawPredictions = RFModel.transform(trainingData);
	Dataset<Row> TrainingRFpredictions = predConverter.transform(labelConverter.transform(TrainingRFrawPredictions));
	//RFpredictions.select("predictionStr", "labelStr", "features").show();
	
	/*************************Random Forest Model Evaluation for Testing Data *********************/
	// View confusion matrix
	System.out.println("Confusion Matrix for Random Forest Tree(training Data):");
	RFpredictions.groupBy(col("labelStr"), col("predictionStr")).count().show();

	// Accuracy computation
	
	
	double TraininigRFaccuracy = evaluator.evaluate(TrainingRFpredictions);
	System.out.println("Random Forest Accuracy = " + Math.round(TraininigRFaccuracy * 100) + " %");
	
	double TrainingRFprecision = evaluator.setMetricName("weightedPrecision").evaluate(TrainingRFpredictions);
	System.out.println("Decision Tree Precision = " + Math.round(TrainingRFprecision*100) + " %");
	
	double TrainingRFrecall = evaluator.setMetricName("weightedRecall").evaluate(TrainingRFpredictions);
	System.out.println("Decision Tree Recall = " + Math.round(TrainingRFrecall*100) + " %");
	

		
		}
}



/* 
 *  Below piece of code was used to find the value for max depth for Decision Tree. After running this we found that 12 will be the best value.
 *  
 *  
for (int i=1; i<30; i++) {
dt.setMaxDepth(i);
DecisionTreeClassificationModel NewModel = dt.fit(trainingData);

Dataset<Row> rawPredictions_check = NewModel.transform(testData);
Dataset<Row> predctions_check = predConverter.transform(labelConverter.transform(rawPredictions_check));

MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
		.setPredictionCol("prediction").setMetricName("accuracy"); 
double accuracy = evaluator.evaluate(predctions_check);
System.out.println("Decision Tree Accuracy for MaxDepth of " +i + "= " + Math.round(accuracy * 100) + " %");

}


Decision Tree Accuracy for MaxDepth of 1= 49 %
Decision Tree Accuracy for MaxDepth of 2= 51 %
Decision Tree Accuracy for MaxDepth of 3= 53 %
Decision Tree Accuracy for MaxDepth of 4= 53 %
Decision Tree Accuracy for MaxDepth of 5= 54 %
Decision Tree Accuracy for MaxDepth of 6= 54 %
Decision Tree Accuracy for MaxDepth of 7= 56 %
Decision Tree Accuracy for MaxDepth of 8= 56 %
Decision Tree Accuracy for MaxDepth of 9= 56 %
Decision Tree Accuracy for MaxDepth of 10= 57 %
Decision Tree Accuracy for MaxDepth of 11= 57 %
Decision Tree Accuracy for MaxDepth of 12= 57 %
Decision Tree Accuracy for MaxDepth of 13= 57 %
Decision Tree Accuracy for MaxDepth of 14= 57 %
Decision Tree Accuracy for MaxDepth of 15= 56 %
Decision Tree Accuracy for MaxDepth of 16= 57 %
Decision Tree Accuracy for MaxDepth of 17= 57 %
Decision Tree Accuracy for MaxDepth of 18= 57 %
Decision Tree Accuracy for MaxDepth of 19= 57 %
Decision Tree Accuracy for MaxDepth of 20= 56 %
Decision Tree Accuracy for MaxDepth of 21= 55 %
Decision Tree Accuracy for MaxDepth of 22= 55 %
Decision Tree Accuracy for MaxDepth of 23= 56 %
Decision Tree Accuracy for MaxDepth of 24= 56 %
Decision Tree Accuracy for MaxDepth of 25= 56 %
Decision Tree Accuracy for MaxDepth of 26= 56 %
Decision Tree Accuracy for MaxDepth of 27= 56 %
Decision Tree Accuracy for MaxDepth of 28= 56 %
Decision Tree Accuracy for MaxDepth of 29= 56 %
Decision Tree Accuracy for MaxDepth of 30= 56 %

 */



/* 
 * Trying to vary the value of Info gain led to reduction in Accuracy
 * Following piece of code was used to test the same
 * for (float i=0.0f; i<=0.4; i=i+0.2f) {
		dt.setMaxDepth(12);
		dt.setMinInfoGain(i);

		//dt.setMinInstancesPerNode(2);
	
		
		DecisionTreeClassificationModel NewModel = dt.fit(trainingData);
		
		Dataset<Row> rawPredictions_check = NewModel.transform(testData);
		Dataset<Row> predctions_check = predConverter.transform(labelConverter.transform(rawPredictions_check));
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy"); 
		double accuracy = evaluator.evaluate(predctions_check);
		System.out.println("Decision Tree Accuracy for Info gain of " +i + "= " + Math.round(accuracy * 100) + " %");

	}
Results

Decision Tree Accuracy for Info gain of 0.0= 57 %
Decision Tree Accuracy for Info gain of 0.2= 41 %

/* 
 * Trying to vary the value of Bins 
 * for (int i=2; i<10; i=i+2) {
		dt.setMaxDepth(12);
		//dt.setMinInfoGain(i);
		dt.setMaxBins(i);

		//dt.setMinInstancesPerNode(2);
	
		
		DecisionTreeClassificationModel NewModel = dt.fit(trainingData);
		
		Dataset<Row> rawPredictions_check = NewModel.transform(testData);
		Dataset<Row> predctions_check = predConverter.transform(labelConverter.transform(rawPredictions_check));
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy"); 
		double accuracy = evaluator.evaluate(predctions_check);
		System.out.println("Decision Tree Accuracy for Max Bins of " +i + "= " + Math.round(accuracy * 100) + " %");

	}
 * With this variation of bins value, we got an increase in accuracy to 58%
 * 
Decision Tree Accuracy for Max Bins of 2= 55 %
Decision Tree Accuracy for Max Bins of 4= 56 %
Decision Tree Accuracy for Max Bins of 6= 58 %
Decision Tree Accuracy for Max Bins of 8= 57 %

 */
 
/*
 * Below piece of code was used to find the max Depth value for Random Classifier algorithm. With a depth of 30, we got an accuracy of 60%
 * for (int i=10; i<=30; i=i+10) {
		RandomForestClassifier rf_new = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setSeed(0);

		rf_new.setMaxDepth(i);
	
		//rf.setNumTrees(10);

		//dt.setMinInstancesPerNode(2);
	
		
		RandomForestClassificationModel RFModelNew = rf_new.fit(trainingData);
		
		Dataset<Row> RFrawPredictions_check = RFModelNew.transform(testData);
		Dataset<Row> predctions_check = predConverter.transform(labelConverter.transform(RFrawPredictions_check));
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy"); 
		double accuracy = evaluator.evaluate(predctions_check);
		System.out.println("Random Forest Classifier Accuracy for Max Depth of " +i + "= " + Math.round(accuracy * 100) + " %");

	}

Results


Random Forest Classifier Accuracy for Max Depth of 10= 58 %
Random Forest Classifier Accuracy for Max Depth of 20= 60 %
Random Forest Classifier Accuracy for Max Depth of 30= 61 %


 */

/*
 * With the help of following piece of code, we tried to find optimal value for number of Trees in Random classifier model
 * for (int i=10; i<=100; i=i+10) {
		RandomForestClassifier rf_new = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setSeed(0);

		rf_new.setMaxDepth(30);
	
		rf_new.setNumTrees(i);

		//dt.setMinInstancesPerNode(2);
	
		
		RandomForestClassificationModel RFModelNew = rf_new.fit(trainingData);
		
		Dataset<Row> RFrawPredictions_check = RFModelNew.transform(testData);
		Dataset<Row> predctions_check = predConverter.transform(labelConverter.transform(RFrawPredictions_check));
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy"); 
		double accuracy = evaluator.evaluate(predctions_check);
		System.out.println("Random Forest Classifier Accuracy for " +i + " number of trees= " + Math.round(accuracy * 100) + " %");

	}

Results:


Random Forest Classifier Accuracy for 10 number of trees= 61 %
Random Forest Classifier Accuracy for 20 number of trees= 61 %
Random Forest Classifier Accuracy for 30 number of trees= 63 %
Random Forest Classifier Accuracy for 40 number of trees= 62 %
Random Forest Classifier Accuracy for 50 number of trees= 63 %
Random Forest Classifier Accuracy for 60 number of trees= 63 %
Random Forest Classifier Accuracy for 70 number of trees= 63 %
Random Forest Classifier Accuracy for 80 number of trees= 62 %
Random Forest Classifier Accuracy for 90 number of trees= 63 %
Random Forest Classifier Accuracy for 100 number of trees= 63 %

*/
