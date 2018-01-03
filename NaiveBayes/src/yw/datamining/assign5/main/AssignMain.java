package yw.datamining.assign5.main;

import java.util.ArrayList;
import java.util.List;

import yw.datamining.assign5.util.Assign5Util;
import yw.datamining.assign5.util.ReadFile;

public class AssignMain {
	
	private String filePath;
	
	private Integer T = 100;
	
	private Integer validNum = 10;
	
	public AssignMain(String filePath) {
		this.filePath = filePath;
	}
	
	public AssignMain(String filePath, Integer T, Integer validNum) {
		this.filePath = filePath;
		this.T = T;
		this.validNum = validNum;
	}
	
	public List<Double> executeAssign5(){
		ReadFile datas = new ReadFile(filePath);
		
		List<Integer> featureType = datas.getFeatureType();
		List<List<Double>> examples = datas.getExamples();
		List<Integer> labels = datas.getLabels();
		for(int i = 0; i < labels.size(); i++){
			if(labels.get(i) == 0){
				labels.set(i, -1);
			}
		}
		
		List<List<Integer>> valid10 = Assign5Util.validDataSet(examples, labels, validNum);
		
		List<Double> resultList = new ArrayList<>();
		List<Double> bayesResult = new ArrayList<>();
		
		for(int ii = 0; ii < this.validNum; ii++){
			List<List<Double>> newExamplesTrain = new ArrayList<>();
			List<Integer> newLabelsTrain = new ArrayList<>();
			
			List<List<Double>> newExamplesTest = new ArrayList<>();
			List<Integer> newLabelsTest = new ArrayList<>();
			
			for(int jj = 0; jj < this.validNum; jj++){
				if(ii != jj){
					valid10.get(jj).forEach(index->{
						newExamplesTrain.add(examples.get(index));
						newLabelsTrain.add(labels.get(index));
					});
				}else{
					valid10.get(jj).forEach(index->{
						newExamplesTest.add(examples.get(index));
						newLabelsTest.add(labels.get(index));
					});
				}
			}
			
			int newTrainNum = newLabelsTrain.size();
			int newTestNum = newLabelsTest.size();
			
			List<Double> px = new ArrayList<>();
			List<Double> HX = new ArrayList<>();
			for(int i = 0; i < newTrainNum; i++){
				px.add(1.0/newTrainNum);
			}
			for(int i = 0; i < newTestNum; i++){
				HX.add(0.0);
			}
			
			for(int i = 0; i < T; i++){
				List<Integer> indexs = Assign5Util.getSubDataSet(newExamplesTrain, px, (int) (newTrainNum*0.9));
				List<List<Double>> subExamples = new ArrayList<>();
				List<Integer> subLabels = new ArrayList<>();
				for(Integer index : indexs){
					subExamples.add(newExamplesTrain.get(index));
					subLabels.add(newLabelsTrain.get(index));
				}
				List<Integer> calcuResult = Assign5Util.classifier(subExamples,newExamplesTrain, subLabels, featureType);
				double errValue = Assign5Util.errorValue(newLabelsTrain, calcuResult);
				bayesResult.add((1 - errValue));
				System.out.println("naive bayes : " + "\t" + (1 - errValue));
				if(errValue > 0.5){
					break;
				}
				double alfa = Math.log((1-errValue)/errValue)/2;
				double tSum = 0;
				for(int j = 0; j < newTrainNum; j++){
					
					double t = 0.0;
					
					if(newLabelsTrain.get(j).compareTo(calcuResult.get(j)) == 0){
						t = px.get(j)*Math.exp(-alfa);
					}else{
						t = px.get(j)*Math.exp(alfa);
					}
					
					px.set(j, t);
					tSum += t;
				}
				
				for(int j = 0; j < newTrainNum; j++){
					px.set(j, px.get(j)/tSum);
				}
				
				List<Integer> calcuResultTest = Assign5Util.classifier(subExamples,newExamplesTest, subLabels, featureType);
				
				for(int j = 0; j < newTestNum; j++){
					HX.set(j, HX.get(j) + calcuResultTest.get(j)*alfa);
				}
			}
			List<Integer> finalResult = new ArrayList<>();
			for(Double tHX : HX){
				if(tHX > 0){
					finalResult.add(1);
				}else{
					finalResult.add(-1);
				}
			}
			Double finalErrValue = Assign5Util.errorValue(newLabelsTest, finalResult);
			System.out.println("adaboost : " + "\t" + (1 - finalErrValue));
			resultList.add(1.0 - finalErrValue);
		}
		
		
		Double bayesMean = AssignMain.mean(bayesResult);
		Double bayesSD = AssignMain.standardDeviation(bayesResult, bayesMean);
		System.out.println(String.format("bayes mean : %f\tSD : %f", bayesMean, bayesSD));
		return resultList;
	}
	
	public static Double mean(List<Double> results){
		Integer size = results.size();
		Double sum = 0.0;
		for(Double result : results){
			sum += result;
		}
		
		return sum/size;
	}
	
	public static Double standardDeviation(List<Double> results, double mean){
		Integer size = results.size();
		Double sum = 0.0;
		for(Double result : results){
			sum += Math.pow(result - mean, 2);
		}
		return Math.pow(sum/size, 0.5);
	}

	public static void main(String[] args) {
		
		AssignMain german = new AssignMain("data/german-assignment5.txt");
		List<Double> germanResult = german.executeAssign5();
		
		Double germanMean = AssignMain.mean(germanResult);
		Double germanSD = AssignMain.standardDeviation(germanResult, germanMean);
		
		System.out.println("german adaboost mean : " + germanMean + "  standardDeviation : " + "\t" + germanSD);
		
		
		AssignMain breast = new AssignMain("data/breast-cancer-assignment5.txt");
		List<Double> breastResult = breast.executeAssign5();
		
		Double breastMean = AssignMain.mean(breastResult);
		Double breastSD = AssignMain.standardDeviation(breastResult, breastMean);
		
		System.out.println("breast adaboost mean : " + breastMean + "standardDeviation : " + "\t" + breastSD);
	}
}
