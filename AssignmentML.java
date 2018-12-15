import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.LogManager;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
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
	//Logger logger = Logger.getRootLogger();
	
	//logger.setLevel(Level.ERROR);
	//logger.getAllAppenders();
		
		SparkSession sparkSession = SparkSession.builder().appName("AssignmentML").master("local[*]").getOrCreate();
		//SparkSession sparkSession = SparkSession.builder().appName("AssignmentML").getOrCreate();
				
		Dataset<Row> TwitterData  = sparkSession.read().option("header", true).option("inferschema", true).csv(args[0]);
		//TwitterData.show();
		//System.out.println(TwitterData.count());
		//TwitterData.printSchema();
		//TwitterData.describe().show();
		Dataset<Row> NewTwitterData = TwitterData.select(col("_unit_id").cast("integer"),col("gender"),
										//col("gender:confidence").cast("float"),col("_golden"),col("gender_gold"),
										col("gender:confidence").cast("float"),
										col("text"),col("retweet_count"),col("link_color"),col("sidebar_color"),
										col("description"), col("fav_number"),
										col("name")); //.toString().length());
		
		NewTwitterData.printSchema();
		System.out.println(NewTwitterData.count());
		
		NewTwitterData.describe().show();
				
		NewTwitterData = NewTwitterData.filter(col("gender:confidence").equalTo("1.0"))
				.filter((col("gender").contains("male")).or(col("gender").contains("brand")).or(col("gender").contains("female")))
				.filter(col("text").isNotNull())
				.filter(col("description").isNotNull());
		System.out.println(NewTwitterData.count());
		NewTwitterData.show();
		//System.exit(0);

		
		StringIndexerModel labelindexer = new StringIndexer()
				.setInputCol("gender")
				.setOutputCol("label").fit(NewTwitterData);
		
		
		Dataset<Row> Gender_label = labelindexer.transform(NewTwitterData);
		Gender_label.show();
		
		NewTwitterData.show();
		//NewTwitterData = NewTwitterData.withColumn("text", regexp_replace(NewTwitterData.col("text"),"[^ 'a-zA-Z0-9,.?!]",""));
		NewTwitterData = NewTwitterData.withColumn("text", regexp_replace(NewTwitterData.col("text"),"<[^>]*>",""));
		NewTwitterData = NewTwitterData.withColumn("emoticons", regexp_extract(NewTwitterData.col("text"),"(\\:\\w+\\:|\\<[\\/\\\\]?3|[\\(\\)\\\\\\D|\\*\\$][\\-\\^]?[\\:\\;\\=]|[\\:\\;\\=B8][\\-\\^]?[3DOPp\\@\\$\\*\\\\\\)\\(\\/\\|])(?=\\s|[\\!\\.\\?]|$)",1));
		
		NewTwitterData = NewTwitterData.withColumn("text", regexp_replace(NewTwitterData.col("text"),"[\\W]"," "));
		NewTwitterData = NewTwitterData.withColumn("text", lower(col("text")));
		
		
		NewTwitterData = NewTwitterData.withColumn("description", regexp_replace(NewTwitterData.col("description"),"[^ 'a-zA-Z0-9,.?!]",""));
		
		NewTwitterData = NewTwitterData.withColumn("description", regexp_replace(NewTwitterData.col("description"),"[\\W]"," "));
		NewTwitterData = NewTwitterData.withColumn("description", lower(col("description")));
		NewTwitterData = NewTwitterData.withColumn("NewText", concat(NewTwitterData.col("text"),NewTwitterData.col("description"), NewTwitterData.col("emoticons")));
		
		
		
		VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"text","description","emoticons","retweet_count","link_color",
                		"sidebar_color","fav_number"})
                .setOutputCol("features");

		//NewTwitterData = NewTwitterData.withColumn("description", )
		
		NewTwitterData.show();
		
		
		// Tokenize the tweet text
				Tokenizer tokenizer = new Tokenizer()
						//.setInputCol("text")
						.setInputCol("NewText")
						//.setInputCol("features")
						.setOutputCol("text_words"); 
		//Stem the words
				Stemmer stem = new Stemmer()
						.setInputCol(tokenizer.getOutputCol())
						.setOutputCol("stem_words").setLanguage("English");
				
	    // Remove the stop words
				StopWordsRemover remover = new StopWordsRemover()
						//.setInputCol(tokenizer.getOutputCol())
						.setInputCol(stem.getOutputCol())
						.setOutputCol("filtered");
		
		// Create the Term Frequency Matrix
				HashingTF hashingTF = new HashingTF()
						.setNumFeatures(100)
						.setInputCol(remover.getOutputCol())
						.setOutputCol("numFeatures");

		// Calculate the Inverse Document Frequency 	
				IDF idf = new IDF()
						.setInputCol(hashingTF.getOutputCol())
						.setOutputCol("features");

	    
			
	    // Set up the Random Forest Model
				RandomForestClassifier rf = new RandomForestClassifier();
				//rf.setNumTrees(30);
			
		//Set up Decision Tree
				DecisionTreeClassifier dt = new DecisionTreeClassifier();

		// Convert indexed labels back to original labels once prediction is available	
				IndexToString labelConverter = new IndexToString()
						.setInputCol("prediction")
						.setOutputCol("predictedLabel").setLabels(labelindexer.labels());

				// Select (prediction, true label) and compute test error.
				MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
						.setLabelCol("label")
						.setPredictionCol("prediction")
						.setMetricName("accuracy")
						.setMetricName("weightedPrecision")
						.setMetricName("weightedRecall");
				
				Dataset<Row>[] splits = NewTwitterData.randomSplit(new double[]{0.7, 0.3},46L);
		        Dataset<Row> trainingData = splits[0];		//Training Data
		        Dataset<Row> testData = splits[1];			//Testing Data
				
				for ( int i= 10; i<=100;i=i+10) {
					// Create and Run Random Forest Pipeline
				rf.setNumTrees(i);
				rf.setSeed(0l);
								
				Pipeline pipelineRF = new Pipeline()
						.setStages(new PipelineStage[] {labelindexer, tokenizer, stem,remover, hashingTF, idf, rf,labelConverter});	
				// Fit the pipeline to training documents.
				//PipelineModel modelRF = pipelineRF.fit(NewTwitterData);	
				PipelineModel modelRF = pipelineRF.fit(trainingData);
				
				// Make predictions on test documents.
				//Dataset<Row> predictionsRF = modelRF.transform(NewTwitterData);
				Dataset<Row> predictionsRF = modelRF.transform(testData);
				System.out.println("Predictions from Random Forest Model are for the value of " +i);
				//predictionsRF.show(10);
				double accuracyRF = evaluator.evaluate(predictionsRF);
				MulticlassMetrics test = new MulticlassMetrics(predictionsRF);
				
				System.out.println("The accuracy is " +test.accuracy());
				System.out.println("The recall values is " +test.weightedRecall());
				System.out.println("Th Precision values is " +test.weightedPrecision());
				
				
				System.out.println("Accuracy = " + Math.round(accuracyRF * 100) + " %");
				System.out.println("Test Error for Random Forest = " + (1.0 - accuracyRF));
				
				}

				// Create and Run Decision Tree Pipeline
				for ( int i= 5; i<=30;i=i+5) {
					// Create and Run Random Forest Pipeline
				dt.setMaxDepth(i);
				dt.setSeed(0L);
				
				Pipeline pipelineDT = new Pipeline()
						.setStages(new PipelineStage[] {labelindexer, tokenizer, stem, remover, hashingTF, idf, dt,labelConverter});	
				// Fit the pipeline to training documents.
				//PipelineModel modelDT = pipelineDT.fit(NewTwitterData);
				PipelineModel modelDT = pipelineDT.fit(trainingData);
				// Make predictions on test documents.
				//Dataset<Row> predictionsDT = modelDT.transform(NewTwitterData);
				Dataset<Row> predictionsDT = modelDT.transform(testData);
				
				System.out.println("Predictions from Decision Tree Model for the value of " +i);
				//predictionsDT.show(10);	
				double accuracyDT = evaluator.evaluate(predictionsDT);
				//double precision = evaluator.evaluate(predictionsDT);
				System.out.println("Accuracy = " + Math.round(accuracyDT * 100) + " %");
				System.out.println("Test Error for Decision Tree = " + (1.0 - accuracyDT));
				
				}
				
				

				//Evaluate Random Forest
				//double accuracyRF = evaluator.evaluate(predictionsRF);
				//System.out.println("Test Error for Random Forest = " + (1.0 - accuracyRF));

				//Evaluate Decision Tree
				
				
	}
}
